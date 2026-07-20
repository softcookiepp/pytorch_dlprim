#if VULKAN_API

#include "CLTensor.h"
#include "utils.h"

#include <dlprim/ops/elementwise.hpp>
#include <dlprim/gpu/program_cache.hpp>
#include <dlprim/json.hpp>
#include <dlprim/utils/json_helpers.hpp>
#include <dlprim/cpu/cpu_ops.hpp>
#include <dlprim/core/pointwise.hpp>

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/ip.hpp>
#include <dlprim/core/bn.hpp>
#include <dlprim/core/conv.hpp>
#include <dlprim/core/interpolate.hpp>
#include <dlprim/core/bias.hpp>
#include <dlprim/core/pool.hpp>
#include <dlprim/gpu/gemm.hpp>
#include <dlprim/gpu/im2col.hpp>

#include <ATen/ops/_native_multi_head_attention_cpu_dispatch.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/DilatedConvolutionUtils.h>
#include <ATen/div_rtn.h>
#include <ATen/native/vol2col.h>

#include <clblast_vk.h>

#include "hvol2col.hpp"

namespace ptdlprim
{

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;

//#include "conv_template.hpp"



#include "conv_template.hpp"

inline void slow_conv2d_shape_check(
		const Tensor& input,
		const Tensor& grad_output,
		const Tensor& weight,
		const Tensor& bias,
		int64_t kernel_height,
		int64_t kernel_width,
		int64_t stride_height,
		int64_t stride_width,
		int64_t pad_height,
		int64_t pad_width,
		bool weight_optional)
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

	if (weight.defined()) {
		TORCH_CHECK(
				weight.numel() > 0 && (weight.dim() == 2 || weight.dim() == 4),
				"non-empty 2D or 4D weight tensor expected, but got: ",
				weight.sizes());
		if (bias.defined()) {
			check_dim_size(bias, 1, 0, weight.size(0));
		}
	} else {
		TORCH_CHECK(weight_optional, "weight tensor is undefined");
	}

	const int64_t ndim = input.dim();
	const int64_t dim_planes = 1;
	const int64_t dim_height = 2;
	const int64_t dim_width = 3;

	// Allow for empty batch size and channel size but not other dimensions
	TORCH_CHECK(ndim == 4, "Expected 4D input tensor, but got: ", input.sizes());
	for (const auto dim : c10::irange(2, ndim)) {
		TORCH_CHECK(input.size(dim) != 0,
								"Expected non-zero size for input dimension ", dim,
								", but got input shape: ", input.sizes(), ". Only the batch and channel dimensions support size 0.");
	}

	const int64_t input_height = input.size(dim_height);
	const int64_t input_width = input.size(dim_width);

	const int64_t exact_input_height = input_height + 2 * pad_height;
	const int64_t exact_input_width = input_width + 2 * pad_width;

	TORCH_CHECK(
			exact_input_height >= kernel_height && exact_input_width >= kernel_width,
			"Calculated padded input size per channel: (",
			exact_input_height,
			" x ",
			exact_input_width,
			"). ",
			"Kernel size: (",
			kernel_height,
			" x ",
			kernel_width,
			"). Kernel size can't be greater than actual input size");

	const int64_t output_height =
			div_rtn<int64_t>(exact_input_height - kernel_height, stride_height) + 1;
	const int64_t output_width =
			div_rtn<int64_t>(exact_input_width - kernel_width, stride_width) + 1;

	TORCH_CHECK(
			output_width >= 1 && output_height >= 1,
			"Given input size per channel: (",
			input_height,
			" x ",
			input_width,
			"). "
			"Calculated output size per channel: (",
			output_height,
			" x ",
			output_width,
			"). Output size is too small");

	if (weight.defined()) {
		int64_t n_input_plane = weight.size(1);
		if (weight.dim() == 2) {
			n_input_plane /= kernel_height;
			n_input_plane /= kernel_width;
		}
		if (input.size(1) != 0) {
			check_dim_size(input, ndim, dim_planes, n_input_plane);
		}
	}

	if (grad_output.defined()) {
		if (weight.defined()) {
			int64_t n_output_plane = weight.size(0);
			check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
		} else if (bias.defined()) {
			TORCH_CHECK(bias.numel() > 0, "non-empty bias tensor expected");
			const int64_t n_output_plane = bias.dim() == 0 ? 1 : bias.size(0);
			check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
		}
		check_dim_size(grad_output, ndim, dim_height, output_height);
		check_dim_size(grad_output, ndim, dim_width, output_width);
	}
}

std::tuple<torch::Tensor&, torch::Tensor&, torch::Tensor&> slow_conv2d_backward_out_vk(
		const torch::Tensor& grad_output,
		const torch::Tensor& self,
		const torch::Tensor& weight,
		torch::IntArrayRef kernel_size,
		torch::IntArrayRef stride,
		torch::IntArrayRef padding,
		torch::Tensor& grad_input,
		torch::Tensor& grad_weight,
		torch::Tensor& grad_bias)
{
	throw std::runtime_error("YOU SHOULD NOT BE HERE EITHER");
	if (grad_input.defined())
	{
#if 0
		slow_conv2d_backward_out_cpu_template(
				grad_input,
				grad_output,
				self,
				weight,
				kernel_size,
				stride,
				padding);
#endif
	}

	if (grad_bias.defined())
	{
		at::sum_out(grad_bias, grad_output, torch::IntArrayRef{0, 2, 3});
	}

	if (grad_weight.defined())
	{

		grad_weight.resize_(weight.sizes(), weight.suggest_memory_format());
		grad_weight.zero_();
#if 0
		slow_conv2d_backward_weight_out_cpu_template(
				grad_weight,
				self,
				grad_output,
				kernel_size,
				stride,
				padding);
#endif
	}

	return std::tuple<torch::Tensor&, torch::Tensor&, torch::Tensor&>(
			grad_input, grad_weight, grad_bias);
}

Tensor slow_conv_dilated2d_vk(
		const Tensor& input,
		const Tensor& weight,
		IntArrayRef kernel_size, const std::optional<Tensor>& bias_opt,
		IntArrayRef stride_size,
		IntArrayRef pad_size,
		IntArrayRef dilation_size)
{
	// See [Note: hacky wrapper removal for optional tensor]
	c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
	const Tensor& bias = *bias_maybe_owned;

	bool use_channels_last = at::native::thnn_conv_use_channels_last(input, weight);
	auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;

	Tensor undefined;
	at::native::internal::slow_conv_dilated_shape_check<2>(
			input,
			weight,
			bias,
			undefined,
			kernel_size,
			stride_size,
			pad_size,
			dilation_size);
	auto is_batch = input.dim() == 4;
	auto options = input.options();
	// calculate output tensor size
	auto output_size = at::native::internal::get_output_size<2>(
			input, weight, kernel_size, stride_size, pad_size, dilation_size);
	// template function assumes batched tensors.	unsqueeze(0) will
	// insert batch dimension without affecting the original tensor.
	const Tensor input_ =
			(is_batch ? input.contiguous(memory_format) : input.contiguous().unsqueeze(0));
	const Tensor weight_ = weight.contiguous(memory_format);
	const Tensor bias_ = (bias.defined() ? bias.contiguous() : undefined);
	Tensor output = at::empty(output_size, options.memory_format(memory_format));
	Tensor output_ = (is_batch ? output : output.unsqueeze(0));

	slow_conv_dilated_all_vk_template(
			output_,
			input_,
			weight_,
			bias_,
			undefined,
			undefined,
			undefined,
			undefined,
			kernel_size,
			stride_size,
			pad_size,
			dilation_size,
			2);

	return output;
}

std::tuple<Tensor, Tensor, Tensor> slow_conv_dilated2d_backward_vk(
		const Tensor& grad_output,
		const Tensor& input,
		const Tensor& weight,
		IntArrayRef kernel_size,
		IntArrayRef stride_size,
		IntArrayRef pad_size,
		IntArrayRef dilation_size,
		const std::array<bool, 3ul> output_mask)
{
	Tensor undefined;
	at::native::internal::slow_conv_dilated_shape_check<2>(
			input,
			weight,
			undefined,
			grad_output,
			kernel_size,
			stride_size,
			pad_size,
			dilation_size);
	auto is_batch = input.dim() == 4;
	auto options = grad_output.options();
	// template function assumes batched tensors.	unsqueeze(0) will
	// insert batch dimension without affecting the original tensor.
	const Tensor grad_output_ =
			(is_batch ? grad_output.contiguous()
								: grad_output.contiguous().unsqueeze(0));
	const Tensor input_ =
			(is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
	const Tensor weight_ = weight.contiguous();
	// compute only gradients for which the corresponding output_mask is true:
	Tensor grad_input =
			(output_mask[0] ? at::empty(input.sizes(), options) : undefined);
	Tensor grad_weight =
			(output_mask[1] ? at::empty(weight.sizes(), options) : undefined);
	Tensor grad_bias =
			(output_mask[2] ? at::empty(weight.size(0), options) : undefined);
	Tensor grad_input_ =
			(output_mask[0] ? (is_batch ? grad_input : grad_input.unsqueeze(0))
											: undefined);
	slow_conv_dilated_all_vk_template(
			undefined,
			input_,
			weight_,
			undefined,
			grad_output_,
			grad_input,
			grad_weight,
			grad_bias,
			kernel_size,
			stride_size,
			pad_size,
			dilation_size,
			2);
	return std::tie(grad_input, grad_weight, grad_bias);
}

std::vector<int64_t> get_conv_transpose2d_output_size(const Tensor& input, const Tensor& weight,
	IntArrayRef kernel_size,
		IntArrayRef stride,
		IntArrayRef padding,
		IntArrayRef output_padding,
		IntArrayRef dilation)
{
	int64_t n_input_plane = weight.size(0);
	int64_t n_output_plane = weight.size(1);

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
	
	int64_t batch_adjust = 0;
	uint batches = 1;
	if (input.sizes().size() == 4)
	{
		// there is a batch
		batch_adjust = 1;
		batches = input.size(0);
	}
	
	int64_t input_height = input.size(1 + batch_adjust);
	int64_t input_width = input.size(2 + batch_adjust);
	int64_t output_height = (input_height - 1) * stride_height - 2 * pad_height +
		(dilation_height * (kernel_height - 1) + 1) + output_padding_height;
	int64_t output_width = (input_width - 1) * stride_width - 2 * pad_width +
		(dilation_width * (kernel_width - 1) + 1) + output_padding_width;
		
	return {batches, n_output_plane, output_height, output_width};
}

Tensor slow_conv_transpose2d_vk(
	const Tensor& input,
	const Tensor& weight,
	IntArrayRef kernel_size,
	//const Tensor& bias_opt,
	const Tensor& bias,
	IntArrayRef stride,
	IntArrayRef padding,
	IntArrayRef output_padding,
	IntArrayRef dilation)
{
	//c10::MaybeOwned<Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias_opt);
	//const Tensor& bias = *bias_maybe_owned;
	std::vector<int64_t> output_size = get_conv_transpose2d_output_size(
		input, weight, kernel_size, stride, padding, output_padding, dilation);
	
	auto options = input.options();
	bool use_channels_last = at::native::thnn_conv_use_channels_last(input, weight);
	auto memory_format = use_channels_last ? at::MemoryFormat::ChannelsLast : at::MemoryFormat::Contiguous;
	Tensor output = at::empty(output_size, options.memory_format(memory_format));
	
	slow_conv_transpose2d_out_vk_template(
			output,
			input,
			weight,
			kernel_size,
			bias,
			stride,
			padding,
			output_padding,
			dilation);
	return output;
}


std::tuple<Tensor, Tensor, Tensor> slow_conv_transpose2d_backward_vk(
		const Tensor& grad_output,
		const Tensor& input,
		const Tensor& weight,
		IntArrayRef kernel_size,
		IntArrayRef stride,
		IntArrayRef padding,
		IntArrayRef output_padding,
		IntArrayRef dilation,
		std::array<bool, 3> output_mask)
{
	Tensor grad_input;
	Tensor grad_weight;
	Tensor grad_bias;

	if (output_mask[0]) {
		grad_input = at::empty({0}, grad_output.options());
	} else {
		grad_input = Tensor();
	}

	if (output_mask[1]) {
		grad_weight = at::empty({0}, grad_output.options());
	} else {
		grad_weight = Tensor();
	}

	if (output_mask[2]) {
		grad_bias = at::empty({0}, grad_output.options());
	} else {
		grad_bias = Tensor();
	}

	if (grad_input.defined())
	{
		slow_conv_transpose2d_backward_out_vk_template(
				input,
				grad_output,
				grad_input,
				weight,
				kernel_size,
				stride,
				padding,
				output_padding,
				dilation);
	}

	if (grad_weight.defined()) {
		grad_weight.resize_(weight.sizes());
		grad_weight.zero_();
	}

	if (grad_bias.defined()) {
		grad_bias.resize_({weight.size(1)});
		grad_bias.zero_();
	}

	if (grad_weight.defined() || grad_bias.defined())
	{
		slow_conv_transpose2d_acc_grad_parameters_vk_template(
				input,
				grad_output,
				grad_weight,
				grad_bias,
				kernel_size,
				stride,
				padding,
				output_padding,
				dilation,
				1);
	}

	return std::tuple<Tensor, Tensor, Tensor>(grad_input, grad_weight, grad_bias);
}

static Tensor _convolution_nogroup_backend(
	const Tensor& input,
	const Tensor& weight,
	const Tensor& bias,
	torch::IntArrayRef stride,
	torch::IntArrayRef padding,
	torch::IntArrayRef output_padding,
	torch::IntArrayRef dilation,
	bool transposed)
		//const ConvBackend backend)
	//	const ConvParams<int64_t>& params)
{
	auto kernel_size = weight.sizes().slice(2);
	size_t dims = input.sizes().size() - 2;
	bool dilated = std::any_of(dilation.cbegin(), dilation.cend(), [](const auto& d) { return d != 1; });
	
	if (dims == 3)
	{
		// 3d
		# if 1
			throw std::runtime_error("conv3d not implemented");
		#else
			if (transposed)
			{
				
			}
			else
			{
				
			}
		#endif
	}
	else if (dims == 2)
	{
		// 2d
		if (transposed)
		{
			return slow_conv_transpose2d_vk(input, weight, kernel_size, bias, stride, padding, output_padding, dilation);
		}
		else if (dilated)
		{
			return slow_conv_dilated2d_vk(input, weight, kernel_size, bias, stride, padding, dilation);
		}
		else
		{
			#if 1
				return slow_conv_dilated2d_vk(input, weight, kernel_size, bias, stride, padding, dilation);
			#else
				return thnn_conv2d_vk(input, weight, kernel_size, bias, params.stride, params.padding);
			#endif
		}
	}
	else
	{
		throw std::runtime_error("invalid dimensions");
	}
#if 0 // just a reference to the original torch code, so we have a guide regarding what to do next!
	switch(backend) {
		case ConvBackend::Slow2d:
			return at::thnn_conv2d(input, weight, kernel_size, bias, params.stride, params.padding);
		case ConvBackend::SlowDilated2d:
			return at::slow_conv_dilated2d(
					input, weight, kernel_size, bias, params.stride, params.padding, params.dilation);
		case ConvBackend::SlowDilated3d:
			return at::slow_conv_dilated3d(
					input, weight, kernel_size, bias, params.stride, params.padding, params.dilation);
		case ConvBackend::SlowTranspose2d:
			return at::slow_conv_transpose2d(
					input, weight, kernel_size, bias, params.stride, params.padding, params.output_padding, params.dilation);
		case ConvBackend::SlowTranspose3d:
			return at::slow_conv_transpose3d(
					input, weight, kernel_size, bias, params.stride, params.padding, params.output_padding, params.dilation);
		default:
			TORCH_CHECK(false, "Unsupported conv nogroup backend encountered");
	}
#endif
}

static std::tuple<at::Tensor, at::Tensor, at::Tensor> _convolution_backward_nogroup_backend(
		const Tensor& grad_output,
		const Tensor& input,
		const Tensor& weight,
		torch::IntArrayRef stride,
		torch::IntArrayRef padding,
		torch::IntArrayRef dilation,
		bool transposed,
		torch::IntArrayRef output_padding,
		const std::array<bool, 3> output_mask)
{
#if 1
	auto kernel_size = weight.sizes().slice(2);
	size_t dims = input.sizes().size() - 2;
	bool dilated = std::any_of(dilation.cbegin(), dilation.cend(), [](const auto& d) { return d != 1; });
	
	// TODO: transposed implicitly supports dilation, modify flow control to suit this
	if(dims == 3)
	{
#if 1
		throw std::runtime_error("Conv3d backward not implemented");
		return std::make_tuple(grad_output, input, weight);
#else
		if (dilated)
		{
			if (transposed)
			{
			}
			else
			{
			}
		}
		else
		{
			if (transposed)
			{
			}
			else
			{
			}
		}
#endif
	}
	else
	{
		if (transposed)
		{
			return slow_conv_transpose2d_backward_vk(
				grad_output, input, weight, kernel_size, stride, padding,
				output_padding, dilation, output_mask);
		}
#if 1
		else
#else
		else if(dilated)
#endif
		{
			return slow_conv_dilated2d_backward_vk(grad_output,
					input,
					weight,
					kernel_size,
					stride,
					padding,
					dilation,
					output_mask);
		}
#if 0
		else
		{
			
		}
#endif
	}
#else
	switch(backend) {
		case ConvBackend::SlowDilated2d:
			return slow_conv_dilated2d_backward_stub(
				input.device().type(),
				grad_output, input, weight, kernel_size, params.stride, params.padding, params.dilation, output_mask);
		case ConvBackend::SlowDilated3d:
			return slow_conv_dilated3d_backward_stub(
				input.device().type(),
				grad_output, input, weight, kernel_size, params.stride, params.padding, params.dilation, output_mask);
		case ConvBackend::SlowTranspose2d:
			return slow_conv_transpose2d_backward_stub(
				input.device().type(), grad_output, input, weight, kernel_size, params.stride, params.padding,
				params.output_padding, params.dilation, output_mask);
		case ConvBackend::SlowTranspose3d:
			return slow_conv_transpose3d_backward_stub(
				input.device().type(), grad_output, input, weight, kernel_size, params.stride, params.padding,
				params.output_padding, params.dilation, output_mask);
		default:
			TORCH_CHECK(false, "Unsupported conv nogroup backend encountered");
	}
#endif
}

static torch::Tensor subtensor(const torch::Tensor& tensor, int64_t dim, int64_t groups, int64_t g)
{
	if (!tensor.defined())
	{
		return torch::Tensor();
	}
	const auto memory_format = tensor.suggest_memory_format();
	int64_t n = tensor.sizes()[dim] / groups;
	return tensor.narrow(dim, n * g, n).contiguous(memory_format);
}

torch::Tensor convolution_overrideable(
	const torch::Tensor & input,
	const torch::Tensor & weight,
	const c10::optional<torch::Tensor> & bias,
	torch::IntArrayRef stride,
	torch::IntArrayRef padding,
	torch::IntArrayRef dilation,
	bool transposed,
	torch::IntArrayRef output_padding,
	int64_t groups)
{
	PTD_TIMER_GUARD("convolution_overrideable");
	GUARD;
	
	c10::MaybeOwned<torch::Tensor> bias_maybe_owned = at::borrow_from_optional_tensor(bias);
	const torch::Tensor& bias_opt = *bias_maybe_owned;
	
	torch::Tensor output;
	auto input_contiguous = input.contiguous();
	auto weight_contiguous = weight.contiguous();
	if (groups == 1)
	{
		output = _convolution_nogroup_backend(input, weight, *bias, stride, padding, output_padding, dilation, transposed);
	}
	else
	{
		std::vector<torch::Tensor> outputs(groups);
		for (size_t g = 0; g < groups; g += 1)
		{
			auto input_g = subtensor(input, 1, groups, g);
			auto weight_g = subtensor(weight, 0, groups, g);
			auto bias_g = subtensor(bias_opt, 0, groups, g);
			outputs[g] = _convolution_nogroup_backend(input_g, weight_g, bias_g, stride, padding, output_padding, dilation, transposed);
		}
		output = at::cat(outputs, 1);
	}
	
	return output;
}

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> convolution_backward_overrideable(
		const torch::Tensor & grad_output,
		const torch::Tensor & input,
		const torch::Tensor & weight,
		torch::IntArrayRef stride,
		torch::IntArrayRef padding,
		torch::IntArrayRef dilation,
		bool transposed,
		torch::IntArrayRef output_padding,
		int64_t groups, std::array<bool,3> output_mask)
{
	PTD_TIMER_GUARD("convolution_backward_overrideable");
	GUARD;
	Tensor backend_grad_input, backend_grad_weight, backend_grad_bias;
	auto kernel_size = weight.sizes().slice(2);
	
	{
		auto input_contiguous = input.contiguous();
		auto weight_contiguous = weight.contiguous();
		if (groups == 1)
		{
			std::tie(backend_grad_input, backend_grad_weight, backend_grad_bias) =
				_convolution_backward_nogroup_backend(
					grad_output, input_contiguous, weight_contiguous, stride, padding, dilation, transposed, output_padding, output_mask);
		}
		else
		{
			std::vector<Tensor> backend_grad_inputs(groups);
			std::vector<Tensor> backend_grad_weights(groups);
			std::vector<Tensor> backend_grad_biases(groups);
			for (int g = 0; g < groups; ++g) {
				auto grad_output_g = subtensor(grad_output, 1, groups, g);
				auto input_g = subtensor(input_contiguous, 1, groups, g);
				auto weight_g = subtensor(weight_contiguous, 0, groups, g);
				std::tie(backend_grad_inputs[g], backend_grad_weights[g], backend_grad_biases[g]) =
					_convolution_backward_nogroup_backend(
						grad_output_g, input_g, weight_g, stride, padding, dilation, transposed, output_padding, output_mask);
			}
			if (output_mask[0]) {
				backend_grad_input = at::cat(backend_grad_inputs, 1);
			}
			if (output_mask[1]) {
				backend_grad_weight = at::cat(backend_grad_weights, 0);
			}
			if (output_mask[2]) {
				backend_grad_bias = at::cat(backend_grad_biases, 0);
			}
		}
	}
	
	return std::make_tuple(std::move(backend_grad_input), std::move(backend_grad_weight), std::move(backend_grad_bias));
}



} // namespace ptdlprim

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
	m.impl("aten::convolution_overrideable",&ptdlprim::convolution_overrideable);
	m.impl("aten::convolution_backward_overrideable",&ptdlprim::convolution_backward_overrideable);
}

#endif
