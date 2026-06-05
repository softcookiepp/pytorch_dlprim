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
#include <ATen/native/ConvUtils.h>
#include <ATen/native/DilatedConvolutionUtils.h>
#include <ATen/div_rtn.h>
#include <ATen/native/vol2col.h>

namespace ptdlprim
{

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;

Tensor& im2col_out_vk(const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride,
    Tensor& output)
{
	throw std::runtime_error("not implemented!");
	return output;
}

Tensor im2col_vk(const Tensor& input,
    IntArrayRef kernel_size,
    IntArrayRef dilation,
    IntArrayRef padding,
    IntArrayRef stride)
{
	Tensor output;
	throw std::runtime_error("not implemented!");
	return output;
}

} // namespace ptdlprim

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
	//m.impl("aten::_slow_conv2d_forward", &ptdlprim::slow_conv2d_forward_vk);
	//m.impl("aten::_slow_conv2d_forward.output", &ptdlprim::slow_conv2d_forward_out_vk);
	m.impl("aten::im2col", &ptdlprim::im2col_vk);
	m.impl("aten::im2col.out", &ptdlprim::im2col_out_vk);
}

#endif
