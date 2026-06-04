
void slow_conv_dilated_location_check(
		const Tensor& input,
		const Tensor& weight,
		const Tensor& bias,
		const Tensor& grad_output)
{
#if 0
	// checking data locations of user-provided tensor arguments
	checkBackend("slow_conv_dilated_location_check", {input, weight}, Backend::CPU);
	if (bias.defined())
	{
		checkBackend("slow_conv_dilated_location_check", {bias}, Backend::CPU);
	}
	if (grad_output.defined())
	{
		checkBackend("slow_conv_dilated_location_check", {grad_output}, Backend::CPU);
	}
#endif
	// we are not checking the data locations of other tensor
	// arguments such as output, grad_input, etc because of these are
	// allocated based on input options and hence these tensors always
	// have the same data location as of input tensor.
}

template <typename Dtype, int64_t dim>
void hvol2col(
		//const Dtype* data_hvol,
		tart::buffer_ptr data_hvol,
		const int channels,
		const IntArrayRef input_size,
		const IntArrayRef output_size,
		const IntArrayRef kernel_size,
		const IntArrayRef stride_size,
		const IntArrayRef pad_size,
		const IntArrayRef dilation_size,
		//Dtype* data_col,
		tart::buffer_ptr data_col,
		bool is_channels_last = false)
{
	if (dim == 3)
	{
#if 1
		throw std::runtime_error("not implemented");
#else
		vol2col<Dtype>(
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
	if (dim == 2)
	{
		if (is_channels_last)
			throw std::runtime_error("channels_last not implemented");
		
		// pretty sure it will be cross-correlation. but I could be totally wrong...
		clblast::Im2col<Dtype>(clblast::KernelMode::kCrossCorrelation, channels, input_size[0], input_size[1],
			kernel_size[0], kernel_size[1], pad_size[0], pad_size[1],
			stride_size[0], stride_size[1], dilation_size[0], dilation_size[1],
			data_hvol, 0, data_col, 0,
			data_hvol->getDevice());
	}
}


template <int64_t dim>
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
		bool is_channels_last = false) {
	slow_conv_dilated_location_check(input, weight, bias, grad_output);
	auto options = input.options();
	// The rear part of input tensor sizes:
	auto input_size = input.sizes().slice(2);
	// The rear part of output tensor sizes:
	auto output_size = at::native::internal::get_output_size<dim>(
			input, kernel_size, stride_size, pad_size, dilation_size);
	int64_t batchSize = input.size(0);
	int64_t nInputPlane = weight.size(1);
	int64_t nOutputPlane = weight.size(0);
	// Temporary buffer:
	Tensor columns = at::empty({0}, options);
	if (output.defined() || grad_weight.defined() || grad_input.defined()) {
		const int64_t m = c10::multiply_integers(kernel_size);
		const int64_t n = c10::multiply_integers(output_size);
		if (is_channels_last) {
			columns.resize_({n, m * nInputPlane});
		} else {
			columns.resize_({nInputPlane * m, n});
		}
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
	// Helpers
	Tensor grad_output_n;
	std::vector<int64_t> dims(dim);
	std::iota(dims.begin(), dims.end(), 1);

		AT_DISPATCH_FLOATING_TYPES_AND3(
				at::ScalarType::Long, at::ScalarType::BFloat16, at::ScalarType::Half, input.scalar_type(), "slow_conv_dilated<>", [&] {
		// For each elt in batch, do:
		for (const auto elt : c10::irange(batchSize)) {
			// Matrix multiply per output:
			Tensor input_n = input.select(0, elt);

			// Output
			if (output.defined()) {
				Tensor output_n = output.select(0, elt);
				if (bias.defined()) {
					/*
						Compute:

							output_n = bias * ones^T

						where

							bias is viewed as bias.view(nOutputPlane, 1)

							ones is viewed as ones.view(outputHeight * outputWidth, 1)

							output_n is viewed as output_n.view(nOutputPlane, outputHeight
					* outputWidth)

					gemm assumes column-major matrices:

						output_n^T = ones * bias^T
						C = alpha * op(A) * op(B)
						op(A) = 't', op(B) = 'n', alpha=1, beta=0
					*/
					// The following for-loop is equivalent to the above
					// gemm setup but avoids allocation of ones tensor:
					for (const auto n : c10::irange(nOutputPlane)) {
						output_n.select(0, n).fill_(bias[n]);
					}
				}
				// Extract columns:
				hvol2col<scalar_t, dim>(
						input_n.const_data_ptr<scalar_t>(),
						nInputPlane,
						input_size,
						output_size,
						kernel_size,
						stride_size,
						pad_size,
						dilation_size,
						columns.mutable_data_ptr<scalar_t>(),
						is_channels_last);
				/*
					Compute:

						output_n = weight * columns + output_n

					where

						weight is viewed as weight.view(nOutputPlane, nInputPlane * kD *
					kH * kW)

						columns size is (nInputPlane * kH * kW) x (outputHeight *
					outputWidth)

						output_n is viewed as output_n.view(nOutputPlane, outputHeight *
					outputWidth)

					gemm assumes column-major matrices:

					channels last:
						output_n^T = weight *columns^T + output_n^T
						C = alpha * op(A) * op(B) + beta * C
						op(A) = 't', op(B) = 'n', alpha=1, beta=1

					channels first:
						output_n^T = columns^T * weight^T + output_n^T
						C = alpha * op(A) * op(B) + beta * C
						op(A) = 'n', op(B) = 'n', alpha=1, beta=1
				*/
				if (is_channels_last)
				{
#if 0
					cpublas::gemm(
							/*transa=*/TransposeType::Transpose,
							/*transb=*/TransposeType::NoTranspose,
							/*		 m=*/nOutputPlane,
							/*		 n=*/columns.size(0),
							/*		 k=*/columns.size(1),
							/* alpha=*/static_cast<scalar_t>(1),
							/*		 A=*/weight.const_data_ptr<scalar_t>(),
							/*	 lda=*/columns.size(1),
							/*		 B=*/columns.const_data_ptr<scalar_t>(),
							/*	 lda=*/columns.size(1),
							/*	beta=*/static_cast<scalar_t>(1),
							/*		 C=*/output_n.mutable_data_ptr<scalar_t>(),
							/*	 ldc=*/nOutputPlane);
#endif
				}
				else
				{
#if 1
					throw std::runtime_error("not implemented");
#else
					cpublas::gemm(
							/*transa=*/TransposeType::NoTranspose,
							/*transb=*/TransposeType::NoTranspose,
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
			} else {
				// All gradients
				grad_output_n = grad_output.select(0, elt);
			}

			// Gradient of input:
			if (grad_input.defined())
			{
				/*
					Compute:

						columns = weight^T * grad_output_n

					where

						weight is viewed as weight.view(nOutputPlane, nInputPlane * kH *
					kW)

						grad_output_n is viewed as grad_output_n.view(nOutputPlane,
					outputHeight * outputWidth)

						columns size is (nInputPlane * kH * kW) x (outputHeight *
					outputWidth)

					gemm assumes column-major matrices:

					channels last:
						columns^T = weight^T * grad_output_n^T
						C = alpha * op(A) * op(B) + beta * C
						op(A) = 'n', op(B) = 'n', alpha=1, beta=0

					channels first:
						columns^T = grad_output_n^T * weight
						C = alpha * op(A) * op(B) + beta * C
						op(A) = 'n', op(B) = 't', alpha=1, beta=0
				 */
				if (is_channels_last) {
					cpublas::gemm(
							/*transa=*/TransposeType::NoTranspose,
							/*transb=*/TransposeType::NoTranspose,
							/*		 m=*/columns.size(1),
							/*		 n=*/columns.size(0),
							/*		 k=*/nOutputPlane,
							/* alpha=*/static_cast<scalar_t>(1),
							/*		 A=*/weight.const_data_ptr<scalar_t>(),
							/*	 lda=*/columns.size(1),
							/*		 B=*/grad_output_n.const_data_ptr<scalar_t>(),
							/*	 ldb=*/nOutputPlane,
							/*	beta=*/static_cast<scalar_t>(0),
							/*		 C=*/columns.mutable_data_ptr<scalar_t>(),
							/*	 ldc=*/columns.size(1));
				} else {
					cpublas::gemm(
							/*transa=*/TransposeType::NoTranspose,
							/*transb=*/TransposeType::Transpose,
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
				}
				// Unpack columns back into input:
				Tensor grad_input_n = grad_input.select(0, elt);

				col2hvol<scalar_t, dim>(
						columns.data_ptr<scalar_t>(),
						nInputPlane,
						input_size,
						output_size,
						kernel_size,
						stride_size,
						pad_size,
						dilation_size,
						grad_input_n.data_ptr<scalar_t>(),
						is_channels_last);
			}

			// Gradient of weight:
			if (grad_weight.defined()) {
				// Extract columns:
				hvol2col<scalar_t, dim>(
						input_n.const_data_ptr<scalar_t>(),
						nInputPlane,
						input_size,
						output_size,
						kernel_size,
						stride_size,
						pad_size,
						dilation_size,
						columns.mutable_data_ptr<scalar_t>(),
						is_channels_last);
				scalar_t scale = 1; // TODO: expose as argument?
				/*
					Compute:

						grad_weight = scale * grad_output_n * columns^T + grad_weight

					where

						grad_output_n is viewed as grad_output_n.view(nOutputPlane,
					outputHeight * outputWidth)

						columns size is (nInputPlane * kD * kH * kW) x (outputHeight *
					outputWidth)

						grad_weight is viewed as grad_weight.view(nOutputPlane,
					nInputPlane * kH * kW)

					gemm assumes column-major matrices:

					channels last:
						grad_weight^T = scale * columns^T * grad_output_n + grad_weight^T
						C = alpha * op(A) * op(B) + beta * C
						op(A) = 'n', op(B) = 't', alpha=scale, beta=1

					channels first:
						grad_weight^T = scale * columns * grad_output_n^T + grad_weight^T
						C = alpha * op(A) * op(B) + beta * C
						op(A) = 't', op(B) = 'n', alpha=scale, beta=1
				*/
				if (is_channels_last) {
					cpublas::gemm(
							/*transa=*/TransposeType::NoTranspose,
							/*transb=*/TransposeType::Transpose,
							/*		 m=*/columns.size(1),
							/*		 n=*/nOutputPlane,
							/*		 k=*/columns.size(0),
							/* alpha=*/static_cast<scalar_t>(scale),
							/*		 A=*/columns.const_data_ptr<scalar_t>(),
							/*	 lda=*/columns.size(1),
							/*		 B=*/grad_output_n.const_data_ptr<scalar_t>(),
							/*	 ldb=*/nOutputPlane,
							/*	beta=*/static_cast<scalar_t>(1),
							/*		 C=*/grad_weight.mutable_data_ptr<scalar_t>(),
							/*	 ldc=*/columns.size(1));
				} else {
					cpublas::gemm(
							/*transa=*/TransposeType::Transpose,
							/*transb=*/TransposeType::NoTranspose,
							/*		 m=*/columns.size(0),
							/*		 n=*/nOutputPlane,
							/*		 k=*/columns.size(1),
							/* alpha=*/static_cast<scalar_t>(scale),
							/*		 A=*/columns.const_data_ptr<scalar_t>(),
							/*	 lda=*/columns.size(1),
							/*		 B=*/grad_output_n.const_data_ptr<scalar_t>(),
							/*	 ldb=*/columns.size(1),
							/*	beta=*/static_cast<scalar_t>(1),
							/*		 C=*/grad_weight.mutable_data_ptr<scalar_t>(),
							/*	 ldc=*/columns.size(0));
				}
			}

			// Gradient of bias:
			if (grad_bias.defined()) {
				/*
					Compute:
						grad_bias = scale * grad_output_n * ones + grad_bias

					where

						grad_bias is viewed as grad_bias.view(nOutputPlane, 1)

						ones is viewed as ones.view(outputHeight * outputWidth, 1)

						grad_output_n is viewed as grad_output_n.view(nOutputPlane,
					outputHeight * outputWidth)

					gemm assumes column-major matrices:

						grad_bias^T = scale * grad_output_n * ones + grad_bias^T
						y = alpha * op(A) * x + beta * y
						op(A) = 't', alpha=scale, beta=1
				 */
				// The following expression is equivalent to the above
				// gemm setup but avoids allocation of ones tensor:
				grad_bias += grad_output_n.sum(dims);
				/*
					TODO: when scale != 1 is introduced then use:
						grad_bias += scale * grad_output_n.sum(dims);
				 */
			}
		}
	});

} // slow_conv_dilated_all_cpu_template

