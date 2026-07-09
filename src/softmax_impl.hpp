#pragma once
#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/loss.hpp>
#include <dlprim/gpu/softmax.hpp>

#include <iostream>

namespace ptdlprim
{
	
using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;

using dlprim::gpu::SoftmaxEpilogue;

Tensor& host_softmax(
	SoftmaxEpilogue epilogue,
	SoftmaxEpilogue epilogueWithMul,
	bool is_log_softmax,
	bool use_fast_softmax,
	const Tensor & input_,
	const int64_t dim_,
	const bool half_to_float,
	Tensor& output);

void host_softmax_backward(
	SoftmaxEpilogue epilogue,
	bool is_log_softmax,
	const Tensor &grad_,
	const Tensor &output_,
	int64_t dim_,
	bool half_to_float,
	const Tensor &gI);

} // namespace ptdlprim
