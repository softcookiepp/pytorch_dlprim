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
#include <dlprim/gpu/im2col.hpp>

#include <ATen/ops/_native_multi_head_attention_cpu_dispatch.h>
#include <ATen/native/ConvUtils.h>
#include <ATen/native/DilatedConvolutionUtils.h>

#include <ATen/native/Fill.h>

#include <ATen/div_rtn.h>
#include <ATen/native/vol2col.h>

#include <clblast_vk.h>


namespace ptdlprim
{

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;



#if 0
// This will eventually be placed in the compute shader, in some way or another.
template<typename scalar_t>
struct FillFunctor {
	FillFunctor(scalar_t v): value(v) {}
	__device__ __forceinline__ scalar_t operator() () const {
		return value;
	}
	private:
		scalar_t value;
};


void fill_vk(TensorIterator& iter, const Scalar& value)
{
	gpu_kernel(iter, FillFunctor<scalar_t>(value.to<scalar_t>()));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ fill ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tensor& fill_out(Tensor& self, const Scalar& value)
{
	if (self.device() == at::kCPU && self.numel() == 1) {
		return at::detail::scalar_fill(self, value);
	}
	auto iter = TensorIteratorConfig()
		.set_check_mem_overlap(false)	// Fill is idempotent, so overlap is okay
		.check_all_same_dtype(false)
		.add_output(self)
		.resize_outputs(false)
		.build();
	fill_stub(iter.device_type(), iter, value);
	return self;
}

#endif

Tensor& fill_Tensor_(Tensor& self, const Tensor& value)
{
	TORCH_CHECK(value.dim() == 0, "fill_ only supports 0-dimension value tensor but got tensor with ", value.dim(), " dimensions.");
	if (self.device() != value.device())
	{
#if 1
		auto valDevice = value.to(self.device());
		return fill_Tensor_(self, valDevice);
#else
		return at::native::fill_out(self, value.item());
#endif
	}
	// Check if value is a view of self and if it is we clone
	// it to avoid overwriting self prematurely
	if(self.is_alias_of(value)) {
		self.copy_(value.clone());
	} else{
		self.copy_(value);
	}
	return self;
}



} // namespace ptdlprim


TORCH_LIBRARY_IMPL(aten, PrivateUse1, m)
{
	m.impl("aten::fill_.Tensor", &ptdlprim::fill_Tensor_);
}

#endif




