#pragma once

void hvol2col(
		const dlprim::ExecutionContext& stream,
		const tart::buffer_ptr& data_hvol,
		const uint32_t data_hvol_offset,
		const int channels,
		const torch::IntArrayRef input_size,
		const torch::IntArrayRef output_size,
		const torch::IntArrayRef kernel_size,
		const torch::IntArrayRef stride_size,
		const torch::IntArrayRef pad_size,
		const torch::IntArrayRef dilation_size,
		const tart::buffer_ptr& data_col,
		const uint32_t data_col_offset,
		const dlprim::DataType Dtype,
		const uint32_t dim)
{
	if (dim == 3)
	{
#if 1
		throw std::runtime_error("not implemented");
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
	else if (dim == 2)
	{
		dlprim::gpu::im2col(
				stream,
				data_hvol,
				data_hvol_offset,
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
				data_col,
				data_col_offset,
				Dtype);
	}
}

void col2hvol(
		const dlprim::ExecutionContext& stream,
		const tart::buffer_ptr& data_col,
		const uint32_t data_col_offset,
		const int channels,
		const torch::IntArrayRef input_size,
		const torch::IntArrayRef output_size,
		const torch::IntArrayRef kernel_size,
		const torch::IntArrayRef stride_size,
		const torch::IntArrayRef pad_size,
		const torch::IntArrayRef dilation_size,
		const tart::buffer_ptr& data_hvol,
		const uint32_t data_hvol_offset,
		const dlprim::DataType Dtype,
		uint32_t dim)
{
	if (dim == 3)
	{
#if 1
		throw std::runtime_error("col2hvol with 3 dims not implemented");
#else
		col2vol<Dtype, Dtype>(
				stream,
				data_col,
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
				data_hvol);
#endif
	}
	if (dim == 2)
	{
		dlprim::gpu::col2im(
			stream,
			data_col,
			data_col_offset,
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
			data_hvol,
			data_hvol_offset,
			Dtype,
			Dtype);
	}
}
