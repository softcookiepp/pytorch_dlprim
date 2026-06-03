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
#include <dlprim/gpu/gemm.hpp>

#include <ATen/ops/_native_multi_head_attention_cpu_dispatch.h>

namespace ptdlprim
{
	
static torch::Tensor _convolution_nogroup_backend(
	const torch::Tensor& input,
    const torch::Tensor& weight,
    const torch::Tensor& bias)
    //const ConvBackend backend)
  //  const ConvParams<int64_t>& params)
{
	throw std::runtime_error("not implemented!");
	torch::Tensor t;
	return t;
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

torch::Tensor convolution_overrideable(const torch::Tensor & input,
                                    const torch::Tensor & weight,
                                    const c10::optional<torch::Tensor> & bias,
                                    torch::IntArrayRef stride,
                                    torch::IntArrayRef padding,
                                    torch::IntArrayRef dilation,
                                    bool transposed,
                                    torch::IntArrayRef output_padding,
                                    int64_t groups)
{
	GUARD;
	
	
	torch::Tensor output;
	auto input_contiguous = input.contiguous();
	auto weight_contiguous = weight.contiguous();
	if (groups == 1)
	{
		output = _convolution_nogroup_backend(input, weight, *bias);
	}
	else
	{
		std::vector<torch::Tensor> outputs(groups);
		for (size_t g = 0; g < groups; g += 1)
		{
			auto input_g = subtensor(input, 1, groups, g);
			auto weight_g = subtensor(weight, 0, groups, g);
			torch::Tensor bias_g;
			if (bias && bias->numel() != 0)
				bias_g = subtensor(*bias, 0, groups, g);
			outputs[g] = _convolution_nogroup_backend(input_g, weight_g, bias_g);
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
	GUARD;
	throw std::runtime_error("not implemented!");
	std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> res;
	return res;
}

} // namespace ptdlprim

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
	m.impl("aten::convolution_overrideable",&ptdlprim::convolution_overrideable);
	m.impl("aten::convolution_backward_overrideable",&ptdlprim::convolution_backward_overrideable);
}

#endif
