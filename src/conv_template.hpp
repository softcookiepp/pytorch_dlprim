
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
				const float beta = 0.0;
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


void slow_conv_transpose2d_out_vk_template(
		const Tensor& output,
		const Tensor& input,
		const Tensor& weight,
		IntArrayRef kernel_size,
		const Tensor& bias,
		IntArrayRef stride,
		IntArrayRef padding,
		IntArrayRef output_padding,
		IntArrayRef dilation)
{
	TensorArg input_arg{input, "input", 1}, output_arg{output, "output", 2},
			weight_arg{weight, "weight", 3}, bias_arg{bias, "bias", 4};
#if 0
	checkAllSameGPU(
			__func__,
			{input_arg, output_arg, weight_arg, bias_arg});
#endif
	int n_input_plane = weight.size(0);
	int n_output_plane = weight.size(1);

	int64_t kernel_height = kernel_size[0];
	int64_t kernel_width = kernel_size[1];
	int64_t dilation_height = dilation[0];
	int64_t dilation_width = dilation[1];
	int64_t pad_height = padding[0];
	int64_t pad_width = padding[1];
	int64_t stride_height = stride[0];
	int64_t stride_width = stride[1];
	int64_t output_padding_height = output_padding[0];
	int64_t output_padding_width = output_padding[1];
	
	dlprim::ExecutionContext stream = getExecutionContext(input);
	dlprim::Context ctx(stream);

	Tensor input_ = input.contiguous();
	Tensor weight_ = weight.contiguous();
	dlprim::Tensor weight_dp = todp(weight_);

	Tensor bias_ = Tensor();

	if (bias.defined()) {
		bias_ = bias.contiguous();
	}

	bool is_batch = false;
	if (input_.dim() == 3) {
		// Force batch
		is_batch = true;
		input_.resize_({1, input_.size(0), input_.size(1), input_.size(2)});
	}

	int64_t input_height = input_.size(2);
	int64_t input_width = input_.size(3);
	int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
			(dilation_height * (kernel_height - 1) + 1) + output_padding_height;
	int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
			(dilation_width * (kernel_width - 1) + 1) + output_padding_width;

	// Batch size + input planes
	int64_t batch_size = input_.size(0);

	// Create temporary columns
	Tensor columns_ = at::empty({n_output_plane * kernel_width * kernel_height,
			input_height * input_width}, input_.options());
	dlprim::Tensor columns_dp = todp(columns_);

	// Define a buffer of ones, for bias accumulation
	Tensor ones_ = bias.defined() ? at::ones({output_height, output_width}, input_.options()) : Tensor();
#if 0
	AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
			input_.scalar_type(), "slow_conv_transpose2d_out_cuda", [&]
#endif
	{
#if 1
		auto accscalar_t = todp(input).dtype();
#else
		using accscalar_t = at::acc_type<scalar_t, true>;
#endif

		// Helpers
		Tensor input_n;
		Tensor output_n;

		// For each elt in batch, do:
		for (int elt = 0; elt < batch_size; elt++) {
			// Matrix multiply per output:
			input_n = input_.select(0, elt);
			output_n = output.select(0, elt);
			
			dlprim::Tensor input_n_dp = todp(input_n);
			dlprim::Tensor output_n_dp = todp(output_n);

			// M,N,K are dims of matrix A and B
			// (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
			int64_t m = weight_.size(1) * weight_.size(2) * weight_.size(3);
			int64_t n = input_height * input_width;
			int64_t k = weight_.size(0);
#if 1
			const float alpha = 1.0;
			const float beta = 0.0;
			clblast::Gemm<float>(clblast::Layout::kColMajor,
				clblast::Transpose::kNo,
				clblast::Transpose::kYes,
				n,
				m,
				k,
				alpha,
				input_n_dp.device_buffer(),
				input_n_dp.device_offset(),
				n,
				weight_dp.device_buffer(),
				weight_dp.device_offset(),
				m,
				beta,
				columns_dp.device_buffer(),
				columns_dp.device_offset(),
				n,
				stream.queue());
#else
			// Do GEMM (note: this is a bit confusing because gemm assumes
			// column-major matrices)
			at::cuda::blas::gemm<scalar_t>(
					'n',
					't',
					n,
					m,
					k,
					1,
					input_n.const_data_ptr<scalar_t>(),
					n,
					weight_.const_data_ptr<scalar_t>(),
					m,
					0,
					columns_.mutable_data_ptr<scalar_t>(),
					n);
#endif

#if 1
			dlprim::gpu::col2im(
				stream,
				columns_dp.device_buffer(),
				columns_dp.device_offset(),
				n_output_plane,
				output_height,
				output_width,
				input_height,
				input_width,
				kernel_height,
				kernel_width,
				pad_height,
				pad_width,
				stride_height,
				stride_width,
				dilation_height,
				dilation_width,
				output_n_dp.device_buffer(),
				output_n_dp.device_offset(),
				output_n_dp.dtype(),
				output_n_dp.dtype());
#else
			// Unpack columns back into input:
			col2im<scalar_t, accscalar_t>(
					at::cuda::getCurrentCUDAStream(),
					columns_.const_data_ptr<scalar_t>(),
					n_output_plane,
					output_height,
					output_width,
					input_height,
					input_width,
					kernel_height,
					kernel_width,
					pad_height,
					pad_width,
					stride_height,
					stride_width,
					dilation_height,
					dilation_width,
					output_n.mutable_data_ptr<scalar_t>());
#endif

			// Do Bias after:
			// M,N,K are dims of matrix A and B
			// (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
			int64_t m_ = n_output_plane;
			int64_t n_ = output_height * output_width;
			int64_t k_ = 1;

			// Do GEMM (note: this is a bit confusing because gemm assumes
			// column-major matrices)
			if (bias.defined())
			{
#if 1
				dlprim::Tensor bias_dp = todp(bias_);
				dlprim::Tensor ones_dp = todp(ones_);
				
				const float alpha = 1.0;
				const float beta = 1.0;
				clblast::Gemm<float>(clblast::Layout::kColMajor,
					clblast::Transpose::kYes,
					clblast::Transpose::kNo,
					n_,
					m_,
					k_,
					alpha,
					ones_dp.device_buffer(),
					ones_dp.device_offset(),
					k_,
					bias_dp.device_buffer(),
					bias_dp.device_offset(),
					k_,
					beta,
					output_n_dp.device_buffer(),
					output_n_dp.device_offset(),
					n_,
					stream.queue());
#else
				at::cuda::blas::gemm<scalar_t>(
						't',
						'n',
						n_,
						m_,
						k_,
						1,
						ones_.const_data_ptr<scalar_t>(),
						k_,
						bias_.const_data_ptr<scalar_t>(),
						k_,
						1,
						output_n.mutable_data_ptr<scalar_t>(),
						n_);
#endif
			}
		}

		// Resize output
		if (is_batch) {
			output.resize_({n_output_plane, output_height, output_width});
			input_.resize_({n_input_plane, input_height, input_width});
		}
	}
#if 0
	); // end of dispatch
#endif
}
