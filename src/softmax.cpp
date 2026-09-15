#if 1
#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/loss.hpp>

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
		dlprim::Tensor x = todp(self, false);
		dlprim::Tensor y = todp(out, false);
		std::vector<int> dims({static_cast<int>(dim)});
		dlprim::core::softmaxAttempt2(x, y, dims, false);
		return out;
	}
	
	// {"schema": "aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
	Tensor & _log_softmax_out(const Tensor & self, int64_t dim, bool half_to_float, Tensor & out)
	{
		dlprim::Tensor x = todp(self, false);
		dlprim::Tensor y = todp(out, false);
		std::vector<int> dims({static_cast<int>(dim)});
		dlprim::core::softmaxAttempt2(x, y, dims, true);
		return out;
	}



	// {"schema": "aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
	Tensor & _log_softmax_backward_data_out(const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType /*input_dtype*/, Tensor & out)
	{
		dlprim::Tensor xGrad = todp(out, false);
		dlprim::Tensor y = todp(output, false);
		dlprim::Tensor yGrad = todp(grad_output, false);
		std::vector<int> dims({static_cast<int>(dim)});
		dlprim::core::softmaxAttempt2Bwd(xGrad, y, yGrad, dims, true);
		return out;
	}

	// {"schema": "aten::_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
	Tensor & _softmax_backward_data_out(const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType /*input_dtype*/, Tensor & grad_input)
	{
		dlprim::Tensor xGrad = todp(grad_input, false);
		dlprim::Tensor y = todp(output, false);
		dlprim::Tensor yGrad = todp(grad_output, false);
		std::vector<int> dims({static_cast<int>(dim)});
		dlprim::core::softmaxAttempt2Bwd(xGrad, y, yGrad, dims, false);
		return grad_input;
	}

} // namespace dlprim
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
	  m.impl("aten::_log_softmax.out",&ptdlprim::_log_softmax_out);
	  m.impl("aten::_log_softmax_backward_data.out",&ptdlprim::_log_softmax_backward_data_out);
	  m.impl("aten::_softmax.out",&ptdlprim::_softmax_out);
	  m.impl("aten::_softmax_backward_data.out",&ptdlprim::_softmax_backward_data_out);;
} 
#endif
