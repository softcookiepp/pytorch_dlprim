#if 1
#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/loss.hpp>
#include <dlprim/gpu/softmax.hpp>
#include "softmax_impl.hpp"

#include <iostream>
namespace ptdlprim {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;


	using torch::Tensor;

	// {"schema": "aten::_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
	Tensor & _softmax_out(const Tensor & self, int64_t dim, bool half_to_float, Tensor & out)
	{
		return host_softmax(
			dlprim::gpu::SoftmaxEpilogue::eForward,
			dlprim::gpu::SoftmaxEpilogue::eForward, // unused as of now, but still provided
			false, false, self, dim, half_to_float, out);
	}

#if 0

	// {"schema": "aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
	Tensor & _log_softmax_backward_data_out(const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType /*input_dtype*/, Tensor & out)
	{
		return impl_softmax_backward_data_out(grad_output,output,dim,true,out);
	}
#endif
	// {"schema": "aten::_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
	Tensor & _softmax_backward_data_out(const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType /*input_dtype*/, Tensor & grad_input)
	{
		std::cout << "uhhhhh" << std::endl;
		return host_softmax_backward(
			SoftmaxEpilogue::eBackward,
			false,
			grad_output,
			output,
			dim,
			false,
			grad_input);
	}

} // namespace dlprim
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
	  // m.impl("aten::_log_softmax.out",&ptdlprim::_log_softmax_out);
	  // m.impl("aten::_log_softmax_backward_data.out",&ptdlprim::_log_softmax_backward_data_out);
	  m.impl("aten::_softmax.out",&ptdlprim::_softmax_out);
	  m.impl("aten::_softmax_backward_data.out",&ptdlprim::_softmax_backward_data_out);;
} 
#endif
