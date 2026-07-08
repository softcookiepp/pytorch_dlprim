
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
					
				// Unpack columns back into input:
				Tensor grad_input_n = grad_input.select(0, elt);
				dlprim::Tensor grad_input_n_dp = todp(grad_input_n);
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
} // slow_conv_dilated_all_vk_template


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
	{
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


			// Do Bias after:
			// M,N,K are dims of matrix A and B
			// (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
			int64_t m_ = n_output_plane;
			int64_t n_ = output_height * output_width;
			int64_t k_ = 1;

			// Do GEMM (note: this is a bit confusing because gemm assumes
			// column-major matrices)
			if (bias.defined() && bias_.defined())
			{
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
			}
		}

		// Resize output
		if (is_batch) {
			output.resize_({n_output_plane, output_height, output_width});
			input_.resize_({n_input_plane, input_height, input_width});
		}
	}
}


static inline void slow_conv_transpose2d_shape_check(
		const Tensor& input,
		const Tensor& grad_output,
		const Tensor& weight,
		const Tensor& bias,
		int kernel_height,
		int kernel_width,
		int stride_height,
		int stride_width,
		int pad_height,
		int pad_width,
		int output_padding_height,
		int output_padding_width,
		int dilation_height,
		int dilation_width,
		bool weight_nullable)
{
	TORCH_CHECK(
			kernel_width > 0 && kernel_height > 0,
			"kernel size should be greater than zero, but got kernel_height: ",
			kernel_height,
			" kernel_width: ",
			kernel_width);
	TORCH_CHECK(
			stride_width > 0 && stride_height > 0,
			"stride should be greater than zero, but got stride_height: ",
			stride_height,
			" stride_width: ",
			stride_width);
	TORCH_CHECK(
			dilation_width > 0 && dilation_height > 0,
			"dilation should be greater than zero, but got dilation_height: ",
			dilation_height,
			", dilation_width: ",
			dilation_width);
	TORCH_CHECK(
			(output_padding_width < stride_width ||
			 output_padding_width < dilation_width) &&
					(output_padding_height < stride_height ||
					 output_padding_height < dilation_height),
			"output padding must be smaller than either stride or dilation, ",
			"but got output_padding_height: ",
			output_padding_height,
			" output_padding_width: ",
			output_padding_width,
			" stride_height: ",
			stride_height,
			" stride_width: ",
			stride_width,
			" dilation_height: ",
			dilation_height,
			" dilation_width: ",
			dilation_width);

	if (weight.defined()) {
		TORCH_CHECK(
				weight.numel() != 0 && (weight.dim() == 2 || weight.dim() == 4),
				"non-empty 2D or 4D weight tensor expected, but got: ",
				weight.sizes());
		if (bias.defined()) {
			check_dim_size(bias, 1, 0, weight.size(1));
		}
	} else if (!weight_nullable) {
		TORCH_CHECK(false, "weight tensor is expected to be non-nullable");
	}

	int ndim = input.dim();
	int dimf = 0;
	int dimh = 1;
	int dimw = 2;

	if (ndim == 4) {
		dimf++;
		dimh++;
		dimw++;
	}

	TORCH_CHECK(
			input.numel() != 0 && (ndim == 3 || ndim == 4),
			"non-empty 3D or 4D input tensor expected but got a tensor with size ",
			input.sizes());

	int64_t input_height = input.size(dimh);
	int64_t input_width = input.size(dimw);
	int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
			(dilation_height * (kernel_height - 1) + 1) + output_padding_height;
	int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
			(dilation_width * (kernel_width - 1) + 1) + output_padding_width;

	if (output_width < 1 || output_height < 1) {
		TORCH_CHECK(false,
				"Given input size per channel: (",
				input_height,
				" x ",
				input_width,
				"). Calculated output spatial size per channel: (",
				output_height,
				" x ",
				output_width,
				"). Output size is too small");
	}

	if (weight.defined()) {
		int64_t n_input_plane = weight.size(0);
		check_dim_size(input, ndim, dimf, n_input_plane);
	}

	if (grad_output.defined()) {
		if (weight.defined()) {
			int64_t n_output_plane = weight.size(1);
			check_dim_size(grad_output, ndim, dimf, n_output_plane);
		} else if (bias.defined()) {
			int64_t n_output_plane = bias.size(0);
			check_dim_size(grad_output, ndim, dimf, n_output_plane);
		}
		check_dim_size(grad_output, ndim, dimh, output_height);
		check_dim_size(grad_output, ndim, dimw, output_width);
	}
}

static void slow_conv_transpose2d_backward_out_vk_template(
		const Tensor& input_,
		const Tensor& grad_output_,
		Tensor& grad_input,
		const Tensor& weight_,
		IntArrayRef kernel_size,
		IntArrayRef stride,
		IntArrayRef padding,
		IntArrayRef output_padding,
		IntArrayRef dilation)
{
	TORCH_CHECK(
			kernel_size.size() == 2,
			"It is expected kernel_size equals to 2, but got size ",
			kernel_size.size());

	TORCH_CHECK(
			dilation.size() == 2,
			"It is expected dilation equals to 2, but got size ",
			dilation.size());

	TORCH_CHECK(
			padding.size() == 2,
			"It is expected padding equals to 2, but got size ",
			padding.size());

	TORCH_CHECK(
			stride.size() == 2,
			"It is expected stride equals to 2, but got size ",
			stride.size());

	TORCH_CHECK(
			output_padding.size() == 2,
			"It is expected stride equals to 2, but got size ",
			output_padding.size());
	
	dlprim::ExecutionContext stream = getExecutionContext(input_);
	dlprim::Context ctx(stream);

	int n_input_plane = weight_.size(0);
	int n_output_plane = weight_.size(1);

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

	slow_conv_transpose2d_shape_check(
			input_,
			grad_output_,
			weight_,
			Tensor(),
			kernel_height,
			kernel_width,
			stride_height,
			stride_width,
			pad_height,
			pad_width,
			output_padding_height,
			output_padding_width,
			dilation_height,
			dilation_width,
			false);

	Tensor input = input_.contiguous();
	Tensor grad_output = grad_output_.contiguous();
	Tensor weight = weight_.contiguous();
	dlprim::Tensor weight_dp = todp(weight);

	bool is_batch = false;
	if (input.dim() == 3) {
		// Force batch
		is_batch = true;
		input.resize_({1, input.size(0), input.size(1), input.size(2)});
		grad_output.resize_(
				{1, grad_output.size(0), grad_output.size(1), grad_output.size(2)});
	}

	int64_t input_width = input.size(3);
	int64_t input_height = input.size(2);
	int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
			(dilation_height * (kernel_height - 1) + 1) + output_padding_height;
	int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
			(dilation_width * (kernel_width - 1) + 1) + output_padding_width;

	// Batch size + input planes
	int64_t batch_size = input.size(0);

	// Resize output
	grad_input.resize_({batch_size, n_input_plane, input_height, input_width});

	// Create temporary columns
	bool need_columns = (kernel_height != 1 || kernel_width != 1 || stride_height != 1 ||
			stride_width != 1 || pad_height != 0 || pad_width != 0 ||
			dilation_height != 1 || dilation_width != 1);
	Tensor grad_columns = need_columns ? at::empty({n_output_plane * kernel_width * kernel_height,
			input_height * input_width}, input.options()) : Tensor();
	dlprim::Tensor grad_columns_dp = todp(grad_columns);

	{
		// Helpers
		Tensor grad_input_n = Tensor();
		Tensor grad_output_n = Tensor();

		// For each elt in batch, do:
		for (int elt = 0; elt < batch_size; elt++)
		{
			// Matrix multiply per sample:
			grad_input_n = grad_input.select(0, elt);
			grad_output_n = grad_output.select(0, elt);
			
			dlprim::Tensor grad_input_n_dp = todp(grad_input_n);
			dlprim::Tensor grad_output_n_dp = todp(grad_output_n);

			if (need_columns)
			{
				dlprim::gpu::im2col(
					stream,
					grad_output_n_dp.device_buffer(),
					grad_output_n_dp.device_offset(),
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
					grad_columns_dp.device_buffer(),
					grad_columns_dp.device_offset(),
					grad_columns_dp.dtype());
			}

			// M,N,K are dims of matrix A and B
			// (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
			int64_t m = weight.size(0);
			int64_t n = input_height * input_width;
			int64_t k = weight.size(1) * weight.size(2) * weight.size(3);

			const float alpha = 1.0;
			const float beta = 0.0;
			
			auto gemm_in_ptr = need_columns ? grad_columns_dp.device_buffer()
					: grad_output_n_dp.device_buffer();
			auto gemm_in_offset = need_columns ? grad_columns_dp.device_offset()
					: grad_output_n_dp.device_offset();
			
			clblast::Gemm(clblast::Layout::kColMajor,
				clblast::Transpose::kNo,
				clblast::Transpose::kNo,
				n,
				m,
				k,
				alpha,
				gemm_in_ptr, gemm_in_offset,
				n,
				weight_dp.device_buffer(), weight_dp.device_offset(),
				k,
				beta,
				grad_input_n_dp.device_buffer(), grad_input_n_dp.device_offset(),
				n,
				stream.queue());
		}

		// Resize output
		if (is_batch) {
			grad_output.resize_({n_output_plane, output_height, output_width});
			input.resize_({n_input_plane, input_height, input_width});
			grad_input.resize_({n_input_plane, input_height, input_width});
		}
	}
}

void slow_conv_transpose2d_acc_grad_parameters_vk_template(
		const Tensor& input_,
		const Tensor& grad_output_,
		Tensor& grad_weight,
		Tensor& grad_bias,
		IntArrayRef kernel_size,
		IntArrayRef stride,
		IntArrayRef padding,
		IntArrayRef output_padding,
		IntArrayRef dilation,
		int scale_)
{
	TORCH_CHECK(
			kernel_size.size() == 2,
			"It is expected kernel_size equals to 2, but got size ",
			kernel_size.size());

	TORCH_CHECK(
			dilation.size() == 2,
			"It is expected dilation equals to 2, but got size ",
			dilation.size());

	TORCH_CHECK(
			padding.size() == 2,
			"It is expected padding equals to 2, but got size ",
			padding.size());

	TORCH_CHECK(
			stride.size() == 2,
			"It is expected stride equals to 2, but got size ",
			stride.size());

	TORCH_CHECK(
			output_padding.size() == 2,
			"It is expected stride equals to 2, but got size ",
			output_padding.size());

	dlprim::ExecutionContext stream = getExecutionContext(input_);
	dlprim::Context ctx(stream);

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

	slow_conv_transpose2d_shape_check(
			input_,
			grad_output_,
			grad_weight,
			grad_bias,
			kernel_height,
			kernel_width,
			stride_height,
			stride_width,
			pad_height,
			pad_width,
			output_padding_height,
			output_padding_width,
			dilation_height,
			dilation_width,
			true);

	Tensor input = input_.contiguous();
	Tensor grad_output = grad_output_.contiguous();

	int64_t n_output_plane;
	if (grad_weight.defined()) {
		n_output_plane = grad_weight.size(1);
	} else if (grad_bias.defined()) {
		n_output_plane = grad_bias.size(0);
	} else {
		return;
	}

	if (grad_weight.defined()) {
		TORCH_CHECK(
				grad_weight.is_contiguous(), "grad_weight needs to be contiguous");
	}

	if (grad_bias.defined()) {
		TORCH_CHECK(grad_bias.is_contiguous(), "grad_bias needs to be contiguous");
	}

	bool is_batch = false;
	if (input.dim() == 3) {
		// Force batch
		is_batch = true;
		input.resize_({1, input.size(0), input.size(1), input.size(2)});
		grad_output.resize_(
				{1, grad_output.size(0), grad_output.size(1), grad_output.size(2)});
	}

	int64_t input_width = input.size(3);
	int64_t input_height = input.size(2);
	int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
			(dilation_height * (kernel_height - 1) + 1) + output_padding_height;
	int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
			(dilation_width * (kernel_width - 1) + 1) + output_padding_width;

	// Batch size + input planes
	int64_t batch_size = input.size(0);

	// Create temporary columns
	bool need_columns = (kernel_height != 1 || kernel_width != 1 || stride_height != 1 ||
			stride_width != 1 || pad_height != 0 || pad_width != 0 ||
			dilation_height != 1 || dilation_width != 1);
	Tensor columns = need_columns ? at::empty({n_output_plane * kernel_width * kernel_height,
			input_height * input_width}, input.options()) : Tensor();

	{
		// Helpers
		Tensor input_n = Tensor();
		
		Tensor grad_output_n = Tensor();
		float scale = scale_;

		// For each elt in batch, do:
		for (int elt = 0; elt < batch_size; elt++)
		{
			// Matrix multiply per output:
			grad_output_n = grad_output.select(0, elt);
			dlprim::Tensor grad_output_n_dp = todp(grad_output_n);

			// Do Weight:
			if (grad_weight.defined())
			{
				// Matrix multiply per output:
				input_n = input.select(0, elt);
				dlprim::Tensor input_n_dp = todp(input_n);
				dlprim::Tensor grad_weight_dp = todp(grad_weight);
				dlprim::Tensor columns_dp = todp(columns);
				if (need_columns)
				{
					dlprim::gpu::im2col(
						stream,
						grad_output_n_dp.device_buffer(),
						grad_output_n_dp.device_offset(),
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
						columns_dp.device_buffer(),
						columns_dp.device_offset(),
						columns_dp.dtype());
				}


				// M,N,K are dims of matrix A and B
				// (see http://docs.nvidia.com/cuda/cublas/#cublas-lt-t-gt-gemm)
				int64_t n = n_output_plane * kernel_height * kernel_width;
				int64_t m = input_n.size(0); // n_input_plane
				int64_t k = input_height * input_width;

				auto gemm_in_ptr = need_columns ? columns_dp.device_buffer()
						: grad_output_n_dp.device_buffer();
						
				auto gemm_in_offset = need_columns ? columns_dp.device_offset()
						: grad_output_n_dp.device_offset();
				
				const float beta = 1.0;
				
				clblast::Gemm(clblast::Layout::kColMajor,
					clblast::Transpose::kYes,
					clblast::Transpose::kNo,
					n,
					m,
					k,
					scale,
					gemm_in_ptr, gemm_in_offset,
					k,
					input_n_dp.device_buffer(), input_n_dp.device_offset(),
					k,
					beta,
					grad_weight_dp.device_buffer(), grad_weight_dp.device_offset(),
					n,
					stream.queue());
			}
		}

		if (grad_bias.defined()) {
			at::sum_out(grad_bias, grad_output, IntArrayRef{0, 2, 3});
		}

		// Resize
		if (is_batch) {
			grad_output.resize_({n_output_plane, output_height, output_width});
			input.resize_({input.size(1), input_height, input_width});
		}
	}
}
