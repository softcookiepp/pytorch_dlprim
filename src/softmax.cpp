
#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/loss.hpp>
#include <dlprim/gpu/softmax.hpp>

#include <iostream>
namespace ptdlprim {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;


	using torch::Tensor;


	static Tensor & impl_softmax_out(const Tensor & self, int64_t dim, bool is_log, Tensor & out)
	{
		GUARD;
		Tensor self_c = self.contiguous();
		dlprim::Tensor x=todp(self_c);
		dlprim::Tensor y=todp(out);
		TORCH_CHECK(dim==-1 || (0<=dim && dim < int(x.shape().size())),"Invalid value of dim");
		if(x.shape().size()!=2) {
			if(dim == -1)
				dim = x.shape().size() - 1;
			int N=1,M=1;
			int Rd = x.shape()[dim];
			for(int i=0;i<dim;i++)
				N*=x.shape()[i];
			for(int i=dim+1;i<int(x.shape().size());i++)
				M*=x.shape()[i];
			auto new_shape = dlprim::Shape(N,Rd,M);
			x.reshape(new_shape);
			y.reshape(new_shape);
		}
		getExecutionContext(self).queue()->sync();
		dlprim::core::softmax_forward(x,y,is_log,getExecutionContext(self));
		sync_if_needed(self.device());
		return out;
	}

	// {"schema": "aten::_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
	Tensor & _softmax_out(const Tensor & self, int64_t dim, bool half_to_float, Tensor & out)
	{
#if 0
#else
		return impl_softmax_out(self,dim,false,out);
#endif
	}

#if 0
	// {"schema": "aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
	Tensor & _log_softmax_out(const Tensor & self, int64_t dim, bool /*half_to_float*/, Tensor & out)
	{
		return impl_softmax_out(self,dim,true,out);
	}

	Tensor & impl_softmax_backward_data_out(const Tensor & grad_output, const Tensor & output, int64_t dim, bool is_log, Tensor & out)
	{
		GUARD;
		dlprim::Tensor dx = todp(out);
		Tensor output_c = output.contiguous(),grad_output_c = grad_output.contiguous();
		dlprim::Tensor y = todp(output_c);
		dlprim::Tensor dy = todp(grad_output_c);
		TORCH_CHECK(dim==-1 || (0<=dim && dim < int(y.shape().size())),"Invalid value of dim");
		if(y.shape().size()!=2) {
			if(dim == -1)
				dim = y.shape().size() - 1;
			int N=1,M=1;
			int Rd = y.shape()[dim];
			for(int i=0;i<dim;i++)
				N*=y.shape()[i];
			for(int i=dim+1;i<int(y.shape().size());i++)
				M*=y.shape()[i];
			auto new_shape = dlprim::Shape(N,Rd,M);
			dx.reshape(new_shape);
			dy.reshape(new_shape);
			y.reshape(new_shape);
		}

		dlprim::core::softmax_backward(dx,y,dy,is_log,0.0f,getExecutionContext(grad_output));
		sync_if_needed(grad_output.device());
		return out;
	}


	// {"schema": "aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) out) -> Tensor(a!)", "dispatch": "True", "default": "False"}
	Tensor & _log_softmax_backward_data_out(const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType /*input_dtype*/, Tensor & out)
	{
		return impl_softmax_backward_data_out(grad_output,output,dim,true,out);
	}

	// {"schema": "aten::_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
	Tensor & _softmax_backward_data_out(const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType /*input_dtype*/, Tensor & grad_input)
	{
		return impl_softmax_backward_data_out(grad_output,output,dim,false,grad_input);
	}
#endif

} // namespace dlprim
#if 0
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
	  // m.impl("aten::_log_softmax.out",&ptdlprim::_log_softmax_out);
	  // m.impl("aten::_log_softmax_backward_data.out",&ptdlprim::_log_softmax_backward_data_out);
	  m.impl("aten::_softmax.out",&ptdlprim::_softmax_out);
	  // m.impl("aten::_softmax_backward_data.out",&ptdlprim::_softmax_backward_data_out);;
} 
#endif
