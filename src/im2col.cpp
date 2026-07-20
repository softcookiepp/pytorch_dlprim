#if VULKAN_API

#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/ip.hpp>
#include <dlprim/core/bn.hpp>
#include <dlprim/core/conv.hpp>
#include <dlprim/core/interpolate.hpp>
#include <dlprim/core/bias.hpp>
#include <dlprim/core/pool.hpp>
#include <dlprim/gpu/im2col.hpp>

#include <ATen/ops/_native_multi_head_attention_cpu_dispatch.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/DilatedConvolutionUtils.h>
#include <ATen/div_rtn.h>
#include <ATen/native/vol2col.h>

#include <ATen/native/im2col.h>
#include <ATen/native/im2col_shape_check.h>
#include <c10/util/irange.h>

namespace ptdlprim
{

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;

static void im2col_out_vk_template(
	Tensor& output,
	const Tensor& input_,
	IntArrayRef kernel_size,
	IntArrayRef dilation,
	IntArrayRef padding,
	IntArrayRef stride)
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

	int64_t kernel_height = kernel_size[0];
	int64_t kernel_width = kernel_size[1];
	int64_t dilation_height = dilation[0];
	int64_t dilation_width = dilation[1];
	int64_t pad_height = padding[0];
	int64_t pad_width = padding[1];
	int64_t stride_height = stride[0];
	int64_t stride_width = stride[1];

	TensorArg input_arg{input_, "input", 1};
	TensorArg output_arg{output, "output", 2};
	
	// no need for this, tart will auto-handle this afaik
	// checkAllSameGPU(__func__, {input_arg, output_arg});

	at::native::im2col_shape_check(
			input_,
			Tensor(),
			kernel_height,
			kernel_width,
			dilation_height,
			dilation_width,
			pad_height,
			pad_width,
			stride_height,
			stride_width);

	Tensor input = input_.contiguous();

	bool batched_input = true;

	if (input.dim() == 3) {
		batched_input = false;
		input = input.unsqueeze(0);
	}

	int64_t batch_size = input.size(0);
	int64_t n_input_plane = input.size(1);
	int64_t input_height = input.size(2);
	int64_t input_width = input.size(3);

	int64_t output_height = (input_height + 2 * pad_height -
													 (dilation_height * (kernel_height - 1) + 1)) /
					stride_height +
			1;
	int64_t output_width = (input_width + 2 * pad_width -
													(dilation_width * (kernel_width - 1) + 1)) /
					stride_width +
			1;
	int64_t n_output_plane = n_input_plane * kernel_width * kernel_height;
	int64_t output_length = output_height * output_width;

	output.resize_({batch_size, n_output_plane, output_length});

	// Launch kernel
	{
		
		Tensor input_n;
		Tensor output_n;
		
		dlprim::ExecutionContext q = getExecutionContext(input);

		for (int64_t elt = 0; elt < batch_size; elt++)
		{
			input_n = input.select(0, elt);
			output_n = output.select(0, elt);
			
			dlprim::Tensor input_n_dp = todp(input_n);
			dlprim::Tensor output_n_dp = todp(output_n);

			dlprim::gpu::im2col(q,
				input_n_dp.device_buffer(),
				input_n_dp.device_offset(),
				n_input_plane,
				input_height,
				input_width,
				output_height,
				output_width,
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
				input_n_dp.dtype());
		}
	}
	
	if (!batched_input)
	{
		output = output.squeeze(0);
	}
}

void col2im_out_vk_template(
		Tensor& output,
		const Tensor& input_,
		IntArrayRef output_size,
		IntArrayRef kernel_size,
		IntArrayRef dilation,
		IntArrayRef padding,
		IntArrayRef stride) {
	TensorArg input_arg{input_, "input", 1};
	TensorArg output_arg{output, "output", 2};
	checkAllSameGPU(__func__, {input_arg, output_arg});

	TORCH_CHECK(
			output_size.size() == 2,
			"It is expected output_size equals to 2, but got size ",
			output_size.size());

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

	int64_t output_height = output_size[0];
	int64_t output_width = output_size[1];
	int64_t kernel_height = kernel_size[0];
	int64_t kernel_width = kernel_size[1];
	int64_t dilation_height = dilation[0];
	int64_t dilation_width = dilation[1];
	int64_t pad_height = padding[0];
	int64_t pad_width = padding[1];
	int64_t stride_height = stride[0];
	int64_t stride_width = stride[1];

	at::native::col2im_shape_check(
		input_,
		Tensor(),
		output_height,
		output_width,
		kernel_height,
		kernel_width,
		dilation_height,
		dilation_width,
		pad_height,
		pad_width,
		stride_height,
		stride_width);

	Tensor input = input_.contiguous();

	bool batched_input = true;
	if (input.dim() == 2) {
		// Force batch
		batched_input = false;
		input = input.unsqueeze(0);
	}

	int64_t batch_size = input.size(0);
	int64_t n_input_plane = input.size(1);
	int64_t n_output_plane = n_input_plane / (kernel_width * kernel_height);
	int64_t input_batch_stride = input.stride(0);

	output.resize_({batch_size, n_output_plane, output_height, output_width});
	int64_t output_batch_stride = output.stride(0);
#if 0
	AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND3(kHalf, kBFloat16, kBool,
			input.scalar_type(), "col2im_out_cuda", [&]
#endif
	{
		int64_t height_col = (output_height + 2 * pad_height -
													(dilation_height * (kernel_height - 1) + 1)) /
						stride_height +
				1;
		int64_t width_col = (output_width + 2 * pad_width -
												 (dilation_width * (kernel_width - 1) + 1)) /
						stride_width +
				1;
#if 1
		//throw std::runtime_error("col2im_batched not implemented!");
		
		dlprim::ExecutionContext q = getExecutionContext(output);
		dlprim::Tensor input_dp = todp(input);
		dlprim::Tensor output_dp = todp(output);
		
		dlprim::gpu::col2im_batched(q,
			input_dp.device_buffer(),
			input_dp.device_offset(),
			input_batch_stride,
			batch_size,
			n_output_plane,
			output_height,
			output_width,
			height_col,
			width_col,
			kernel_height,
			kernel_width,
			pad_height,
			pad_width,
			stride_height,
			stride_width,
			dilation_height,
			dilation_width,
			output_dp.device_buffer(),
			output_dp.device_offset(),
			output_batch_stride,
			output_dp.dtype());
		
#else
		col2im_batched(
				at::cuda::getCurrentCUDAStream(),
				input.const_data_ptr<scalar_t>(),
				input_batch_stride,
				batch_size,
				n_output_plane,
				output_height,
				output_width,
				height_col,
				width_col,
				kernel_height,
				kernel_width,
				pad_height,
				pad_width,
				stride_height,
				stride_width,
				dilation_height,
				dilation_width,
				output.mutable_data_ptr<scalar_t>(),
				output_batch_stride);
#endif
	}
#if 0
	);
#endif
	if (!batched_input) {
		output = output.squeeze(0);
	}
}

Tensor& im2col_out_vk(const Tensor& input,
		IntArrayRef kernel_size,
		IntArrayRef dilation,
		IntArrayRef padding,
		IntArrayRef stride,
		Tensor& output)
{
	PTD_TIMER_GUARD("im2col_out_vk");
	im2col_out_vk_template(output, input, kernel_size, dilation, padding, stride);
	return output;
}

Tensor im2col_vk(const Tensor& input,
		IntArrayRef kernel_size,
		IntArrayRef dilation,
		IntArrayRef padding,
		IntArrayRef stride)
{
	Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
	im2col_out_vk(input, kernel_size, dilation, padding, stride, output);
	return output;
}

Tensor& col2im_out_vk(const Tensor& input,
		IntArrayRef output_size,
		IntArrayRef kernel_size,
		IntArrayRef dilation,
		IntArrayRef padding,
		IntArrayRef stride,
		Tensor& output)
{
	PTD_TIMER_GUARD("col2im_out_vk");
	col2im_out_vk_template(
			output, input, output_size, kernel_size, dilation, padding, stride);
	return output;
}

Tensor col2im_vk(
		const Tensor& input,
		IntArrayRef output_size,
		IntArrayRef kernel_size,
		IntArrayRef dilation,
		IntArrayRef padding,
		IntArrayRef stride)
{
	Tensor output = at::empty_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
	col2im_out_vk_template(
			output, input, output_size, kernel_size, dilation, padding, stride);
	return output;
}

} // namespace ptdlprim

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
	//m.impl("aten::_slow_conv2d_forward", &ptdlprim::slow_conv2d_forward_vk);
	//m.impl("aten::_slow_conv2d_forward.output", &ptdlprim::slow_conv2d_forward_out_vk);
	m.impl("aten::im2col", &ptdlprim::im2col_vk);
	m.impl("aten::im2col.out", &ptdlprim::im2col_out_vk);
	m.impl("aten::col2im", &ptdlprim::col2im_vk);
	m.impl("aten::col2im.out", &ptdlprim::col2im_out_vk);
}

#endif
