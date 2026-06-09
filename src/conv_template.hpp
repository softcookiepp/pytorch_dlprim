
std::vector<int64_t> get_output_size(
		const at::Tensor& input,
		c10::IntArrayRef& kernel_size,
		c10::IntArrayRef& stride_size,
		c10::IntArrayRef& pad_size,
		c10::IntArrayRef& dilation_size,
		int64_t dim) {
	std::vector<int64_t> sizes;
	for (const auto index : c10::irange(dim)) {
		sizes.push_back(
				div_rtn<int64_t>(
						input.size(index + input.dim() - dim) + 2 * pad_size[index] -
								(dilation_size[index] * (kernel_size[index] - 1) + 1),
						stride_size[index]) +
				1);
	}
	return sizes;
}

void slow_conv_dilated_all_vk_template(
		Tensor& output,
		const Tensor& input,
		const Tensor& weight,
		const Tensor& bias,
		const Tensor& grad_output,
		Tensor& grad_input,
		Tensor& grad_weight,
		Tensor& grad_bias,
		IntArrayRef kernel_size,
		IntArrayRef stride_size,
		IntArrayRef pad_size,
		IntArrayRef dilation_size,
		const uint32_t dim)
{
	// not needed; tart takes care of that
	//slow_conv_dilated_location_check(__func__, input, weight, bias, grad_output);
	dlprim::ExecutionContext stream = getExecutionContext(input);
	dlprim::Context ctx(stream);
	
	auto options = input.options();
	// The rear part of input tensor sizes:
	auto input_size = input.sizes().slice(2);
	// The rear part of output tensor sizes:
	auto output_size = get_output_size(input, kernel_size, stride_size, pad_size, dilation_size, dim);
	int64_t batchSize = input.size(0);
	int64_t nInputPlane = weight.size(1);
	int64_t nOutputPlane = weight.size(0);
	// Temporary buffers:
	const int64_t m = c10::multiply_integers(kernel_size);
	const int64_t output_vsize = c10::multiply_integers(output_size);
	Tensor columns = at::empty({0}, options);
	if (output.defined() || grad_weight.defined() || grad_input.defined()) {
		columns.resize_({nInputPlane * m, output_vsize});
	}
	// Initialize
	if (grad_weight.defined()) {
		grad_weight.zero_();
	}
	if (grad_bias.defined()) {
		grad_bias.zero_();
	}
	if (output.defined() && !bias.defined()) {
		output.zero_();
	}
// not sure if this will be needed; time will tell...
#if 0 //defined(USE_ROCM)
	/* When using ROCm, the sum evaluation is inaccurate for double
		 tensors. The reason is currently unknown. Hence, we use gemv for
		 computing `grad_output_n.sum(dims)` until the ROCm-sum issue is
		 resolved. */
	Tensor ones = at::empty({0}, options);
	if (grad_bias.defined()) {
		ones.resize_({output_vsize});
		ones.fill_(1);
	}
	/* MSVC does not like #ifdef-s inside the CPP macro
		 AT_DISPATCH_FLOATING_TYPES_AND_HALF. So, we define the code
		 branching outside the CPP macro: */
#define CALCULATE_GRAD_BIAS																\
	at::cuda::blas::gemv<scalar_t>(													\
			/*trans=*/'t',																			 \
			/*		m=*/output_vsize,															\
			/*		n=*/nOutputPlane,															\
			/*alpha=*/static_cast<scalar_t>(1),									\
			/*		A=*/grad_output_n.const_data_ptr<scalar_t>(),	\
			/*	lda=*/output_vsize,															\
			/*		x=*/ones.const_data_ptr<scalar_t>(),					 \
			/* incx=*/1,																				 \
			/* beta=*/static_cast<scalar_t>(1),									\
			/*		y=*/grad_bias.mutable_data_ptr<scalar_t>(),		\
			/* incy=*/1)
#else
#define CALCULATE_GRAD_BIAS grad_bias += grad_output_n.sum(dims)
#endif

	// Helpers
	Tensor grad_output_n;
	std::vector<int64_t> dims(dim);
	std::iota(dims.begin(), dims.end(), 1);
	
	dlprim::Tensor columns_dp = todp(columns);
	dlprim::Tensor weight_dp = todp(weight);
#if 0
	AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
			input.scalar_type(), "slow_conv_dilated<>", [&]
#endif
	{
		// For each elt in batch, do:
		for (int elt = 0; elt < batchSize; elt++)
		{
			// Matrix multiply per output:
			Tensor input_n = input.select(0, elt);
			dlprim::Tensor input_n_dp = todp(input_n);

			// Output
			if (output.defined())
			{
				Tensor output_n = output.select(0, elt);
				if (bias.defined()) {
					/* For gemm argument derivation, see
						 slow_conv_dilated_all_cuda_template in
						 ATen/native/DilatedConvolution.cpp */
					for (int n = 0; n < nOutputPlane; n++) {
						output_n.select(0, n).fill_(bias[n]);
					}
				}
				dlprim::Tensor output_n_dp = todp(output_n);
				
				// Extract columns:
				hvol2col(
					stream,
					input_n_dp.device_buffer(),
					input_n_dp.device_offset(),
					nInputPlane,
					input_size,
					output_size,
					kernel_size,
					stride_size,
					pad_size,
					dilation_size,
					columns_dp.device_buffer(),
					columns_dp.device_offset(),
					input_n_dp.dtype(),
					dim);
#if 1
				const float alpha = 1.0;
				const float beta = 1.0;
				clblast::Gemm(clblast::Layout::kColMajor, clblast::Transpose::kNo, clblast::Transpose::kNo,
					columns.size(1),
					nOutputPlane,
					columns.size(0),
					alpha,
					columns_dp.device_buffer(), columns_dp.device_offset(), columns.size(1),
					weight_dp.device_buffer(), weight_dp.device_offset(), columns.size(0),
					beta,
					output_n_dp.device_buffer(), output_n_dp.device_offset(), columns.size(1),
					stream.queue());
#else
				/* For gemm argument derivation, see
					 slow_conv_dilated_all_cuda_template in
					 ATen/native/DilatedConvolution.cpp */
				at::cuda::blas::gemm<scalar_t>(
						/*transa=*/'n',
						/*transb=*/'n',
						/*		 m=*/columns.size(1),
						/*		 n=*/nOutputPlane,
						/*		 k=*/columns.size(0),
						/* alpha=*/static_cast<scalar_t>(1),
						/*		 A=*/columns.const_data_ptr<scalar_t>(),
						/*	 lda=*/columns.size(1),
						/*		 B=*/weight.const_data_ptr<scalar_t>(),
						/*	 ldb=*/columns.size(0),
						/*	beta=*/static_cast<scalar_t>(1),
						/*		 C=*/output_n.mutable_data_ptr<scalar_t>(),
						/*	 ldc=*/columns.size(1));
#endif

			}
			else
			{
				// All gradients
				grad_output_n = grad_output.select(0, elt);
			}
			
			dlprim::Tensor grad_output_n_dp;
			
			if (grad_output_n.defined())
			{
				grad_output_n_dp = todp(grad_output_n);
			}

			// Gradient of input:
			if (grad_input.defined())
			{
#if 1
				const float alpha = 1.0;
				const float beta = 1.0;
				clblast::Gemm(clblast::Layout::kColMajor, clblast::Transpose::kNo, clblast::Transpose::kYes,
					columns.size(1),
					columns.size(0),
					nOutputPlane,
					alpha,
					grad_output_n_dp.device_buffer(), grad_output_n_dp.device_offset(), columns.size(1), // a
					weight_dp.device_buffer(), weight_dp.device_offset(), columns.size(0), // b
					beta,
					columns_dp.device_buffer(), columns_dp.device_offset(), columns.size(1), // c
					stream.queue());
#else
				/* For gemm argument derivation, see
					 slow_conv_dilated_all_cuda_template in
					 ATen/native/DilatedConvolution.cpp */
				at::cuda::blas::gemm<scalar_t>(
						/*transa=*/'n',
						/*transb=*/'t',
						/*		 m=*/columns.size(1),
						/*		 n=*/columns.size(0),
						/*		 k=*/nOutputPlane,
						/* alpha=*/static_cast<scalar_t>(1),
						/*		 A=*/grad_output_n.const_data_ptr<scalar_t>(),
						/*	 lda=*/columns.size(1),
						/*		 B=*/weight.const_data_ptr<scalar_t>(),
						/*	 ldb=*/columns.size(0),
						/*	beta=*/static_cast<scalar_t>(0),
						/*		 C=*/columns.mutable_data_ptr<scalar_t>(),
						/*	 ldc=*/columns.size(1));
#endif
				// Unpack columns back into input:
				Tensor grad_input_n = grad_input.select(0, elt);
				dlprim::Tensor grad_input_n_dp = todp(grad_input_n);
#if 1
				col2hvol(
						stream,
						columns_dp.device_buffer(),
						columns_dp.device_offset(),
						nInputPlane,
						input_size,
						output_size,
						kernel_size,
						stride_size,
						pad_size,
						dilation_size,
						grad_input_n_dp.device_buffer(),
						grad_input_n_dp.device_offset(),
						grad_input_n_dp.dtype(),
						dim);
#else
				col2hvol<scalar_t, dim>(
						stream,
						columns.const_data_ptr<scalar_t>(),
						nInputPlane,
						input_size,
						output_size,
						kernel_size,
						stride_size,
						pad_size,
						dilation_size,
						grad_input_n.mutable_data_ptr<scalar_t>());
#endif
			}

			// Gradient of weight:
			if (grad_weight.defined()) {
				// Extract columns:
				hvol2col(
					stream,
					input_n_dp.device_buffer(),
					input_n_dp.device_offset(),
					nInputPlane,
					input_size,
					output_size,
					kernel_size,
					stride_size,
					pad_size,
					dilation_size,
					columns_dp.device_buffer(),
					columns_dp.device_offset(),
					columns_dp.dtype(),
					dim);
#if 1
				const float alpha = 1.0;
				const float beta = 1.0;
				
				dlprim::Tensor grad_weight_dp = todp(grad_weight);
				
				clblast::Gemm(clblast::Layout::kColMajor, clblast::Transpose::kYes, clblast::Transpose::kNo,
					columns.size(0),
					nOutputPlane,
					columns.size(1),
					alpha,
					columns_dp.device_buffer(), columns_dp.device_offset(), columns.size(1), // a
					grad_output_n_dp.device_buffer(), grad_output_n_dp.device_offset(), columns.size(1), // b
					beta,
					grad_weight_dp.device_buffer(), grad_weight_dp.device_offset(), columns.size(0), // c
					stream.queue());
#else
				scalar_t scale = static_cast<scalar_t>(
						1); // TODO: expose as argument?
				/* For gemm argument derivation, see
					 slow_conv_dilated_all_cuda_template in
					 ATen/native/DilatedConvolution.cpp */

				at::cuda::blas::gemm<scalar_t>(
						/*transa=*/'t',
						/*transb=*/'n',
						/*		 m=*/columns.size(0),
						/*		 n=*/nOutputPlane,
						/*		 k=*/columns.size(1),
						/* alpha=*/scale,
						/*		 A=*/columns.const_data_ptr<scalar_t>(),
						/*	 lda=*/columns.size(1),
						/*		 B=*/grad_output_n.const_data_ptr<scalar_t>(),
						/*	 ldb=*/columns.size(1),
						/*	beta=*/static_cast<scalar_t>(1),
						/*		 C=*/grad_weight.mutable_data_ptr<scalar_t>(),
						/*	 ldc=*/columns.size(0));
#endif
			}

			// Gradient of bias:
			if (grad_bias.defined()) {
				/* For gemv argument derivation, see
					 slow_conv_dilated_all_cpu_template in
					 ATen/native/DilatedConvolution.cpp */
				CALCULATE_GRAD_BIAS; /* MSVC does not like #ifdef-s
																inside the CPP macros, see above. */
				/*
					TODO: when scale != 1 is introduced then use:
						grad_bias += scale * grad_output_n.sum(dims);
				 */
			}
		}
	}
#if 0
	);
#endif

} // slow_conv_dilated_all_cuda_template
