
template <typename dt>
void im2col(
		const tart::device_ptr& stream,
		const tart::buffer_ptr& data_im,
		const int64_t channels,
		const int64_t height,
		const int64_t width,
		const int64_t height_col,
		const int64_t width_col,
		const int64_t kernel_height,
		const int64_t kernel_width,
		const int64_t pad_height,
		const int64_t pad_width,
		const int64_t stride_height,
		const int64_t stride_width,
		const int64_t dilation_height,
		const int64_t dilation_width,
		const tart::buffer_ptr& data_col) {
	// We are going to launch channels * height_col * width_col kernels, each
	// kernel responsible for copying a single-channel grid.
	int64_t num_kernels = channels * height_col * width_col;
	// Launch CUDA_NUM_THREADS = 1024
	im2col_kernel<<<GET_BLOCKS(num_kernels), 1024, 0, stream>>>(
			num_kernels,
			data_im,
			height,
			width,
			kernel_height,
			kernel_width,
			pad_height,
			pad_width,
			stride_height,
			stride_width,
			dilation_height,
			dilation_width,
			height_col,
			width_col,
			data_col);
	//C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// hyper-volume to column, CUDA
template <typename Dtype, int64_t dim>
void hvol2col(
		const tart::device_ptr stream,
		const Dtype* data_hvol,
		const int channels,
		const IntArrayRef input_size,
		const IntArrayRef output_size,
		const IntArrayRef kernel_size,
		const IntArrayRef stride_size,
		const IntArrayRef pad_size,
		const IntArrayRef dilation_size,
		Dtype* data_col) {
	if (dim == 3) {
#if 1
		throw std::runtime_error("not implemented!");
#else
		vol2col<Dtype>(
				stream,
				data_hvol,
				channels,
				input_size[0],
				input_size[1],
				input_size[2],
				output_size[0],
				output_size[1],
				output_size[2],
				kernel_size[0],
				kernel_size[1],
				kernel_size[2],
				pad_size[0],
				pad_size[1],
				pad_size[2],
				stride_size[0],
				stride_size[1],
				stride_size[2],
				dilation_size[0],
				dilation_size[1],
				dilation_size[2],
				data_col);
#endif
	}
	if (dim == 2) {
		im2col<Dtype>(
				stream,
				data_hvol,
				channels,
				input_size[0],
				input_size[1],
				output_size[0],
				output_size[1],
				kernel_size[0],
				kernel_size[1],
				pad_size[0],
				pad_size[1],
				stride_size[0],
				stride_size[1],
				dilation_size[0],
				dilation_size[1],
				data_col);
	}
}

template <int64_t dim>
void slow_conv_dilated_all_cuda_template(
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
		IntArrayRef dilation_size) {
	slow_conv_dilated_location_check(__func__, input, weight, bias, grad_output);
	
	dlprim::Tensor dpInput = todp(input);
	dlprim::Tensor dpWeight = todp(weight);
	dlprim::Tensor dpBias = todp(bias);
	dlprim::Tensor dpGradOutput = todp(grad_output);

	// cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	tart::device_ptr stream = dpInput.device_buffer()->getDevice();
	
	auto options = input.options();
	// The rear part of input tensor sizes:
	auto input_size = input.sizes().slice(2);
	// The rear part of output tensor sizes:
	auto output_size = internal::get_output_size<dim>(
			input, kernel_size, stride_size, pad_size, dilation_size);
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

#if 0//defined(USE_ROCM)
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

	AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16,
			input.scalar_type(), "slow_conv_dilated<>", [&] {
				// For each elt in batch, do:
				for (int elt = 0; elt < batchSize; elt++) {
					// Matrix multiply per output:
					Tensor input_n = input.select(0, elt);

					// Output
					if (output.defined()) {
						Tensor output_n = output.select(0, elt);
						if (bias.defined()) {
							/* For gemm argument derivation, see
								 slow_conv_dilated_all_cuda_template in
								 ATen/native/DilatedConvolution.cpp */
							for (int n = 0; n < nOutputPlane; n++) {
								output_n.select(0, n).fill_(bias[n]);
							}
						}
						// Extract columns:
						hvol2col<scalar_t, dim>(
								stream,
								input_n.const_data_ptr<scalar_t>(),
								nInputPlane,
								input_size,
								output_size,
								kernel_size,
								stride_size,
								pad_size,
								dilation_size,
								columns.mutable_data_ptr<scalar_t>());
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

					} else {
						// All gradients
						grad_output_n = grad_output.select(0, elt);
					}

					// Gradient of input:
					if (grad_input.defined()) {
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
						// Unpack columns back into input:
						Tensor grad_input_n = grad_input.select(0, elt);

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
					}

					// Gradient of weight:
					if (grad_weight.defined()) {
						// Extract columns:
						hvol2col<scalar_t, dim>(
								stream,
								input_n.const_data_ptr<scalar_t>(),
								nInputPlane,
								input_size,
								output_size,
								kernel_size,
								stride_size,
								pad_size,
								dilation_size,
								columns.mutable_data_ptr<scalar_t>());
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
			});

} // slow_conv_dilated_all_cuda_template
