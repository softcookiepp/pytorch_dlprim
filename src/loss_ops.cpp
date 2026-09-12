#include "CLTensor.h"
#include "utils.h"

#include <dlprim/core/util.hpp>
#include <dlprim/core/pointwise.hpp>
#include <dlprim/core/loss.hpp>

#include <iostream>

#include "softmax_impl.hpp"

namespace ptdlprim {

using namespace torch;
using torch::autograd::tensor_list;
using torch::autograd::AutogradContext;


using c10::Device;
using c10::DeviceType;


	using torch::Tensor;

	// {"schema": "aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))", "dispatch": "True", "default": "False"}
	::std::tuple<Tensor &,Tensor &> nll_loss_forward_out(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, Tensor & output, Tensor & total_weight)
	{
		GUARD;
		TORCH_CHECK(!weight || weight->numel()==0,"Weight NLLLoss isn't supported");
		TORCH_CHECK(ignore_index <0,"Ignore index isn't supported");
		Tensor self_c = self.contiguous();
		dlprim::Tensor x=todp(self_c);
		Tensor target_c = target.contiguous();
		dlprim::Tensor lbl=todp(target_c);
		dlprim::Tensor y=todp(output);
		bool reduce = false;
		float scale = 1;
		switch(reduction) {
		case 0: reduce=false; break; // None
		case 1: reduce=true; scale = 1.0f/x.shape()[0]; break; // Mean
		case 2: reduce=true; break; // sum
		}
		dlprim::core::nll_loss_forward(x,lbl,y,reduce,scale);
		sync_if_needed(self.device());
		return std::tuple<Tensor &,Tensor &>(output,total_weight);
	}

	// {"schema": "aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, int ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
	Tensor & nll_loss_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, int64_t ignore_index, const Tensor & /*total_weight*/, Tensor & grad_input)
	{
		GUARD;
		TORCH_CHECK(!weight || weight->numel()==0,"Weight NLLLoss isn't supported");
		TORCH_CHECK(ignore_index <0,"Ignore index isn't supported");
		dlprim::Tensor dx=todp(grad_input);
		Tensor target_c = target.contiguous(), grad_output_c = grad_output.contiguous();
		dlprim::Tensor lbl=todp(target_c);
		dlprim::Tensor dy=todp(grad_output_c);
		bool reduce = false;
		float scale = 1;
		switch(reduction) {
		case 0: reduce=false; break; // None
		case 1: reduce=true; scale = 1.0f/dx.shape()[0]; break; // Mean
		case 2: reduce=true; break; // sum
		}
		dlprim::core::nll_loss_backward(dx,lbl,dy,reduce,scale,0.0f);
		sync_if_needed(self.device());
		return grad_input;
	}
	   
	 // {"schema": "aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor", "dispatch": "True", "default": "False"}
	Tensor binary_cross_entropy(const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction)
	{
		GUARD;
		TORCH_CHECK(!weight || weight->numel()==0,"Weight in binar_cross_entroy isn't supported");
		dlprim::Tensor x = todp(self, true);
		dlprim::Tensor y = todp(target, true);
		float scale = 1.0;
		bool reduce = false;
		switch(reduction)
		{
			case 0: break;
			case 1: reduce = true; scale = scale = 1.0/x.shape().total_size(); break; // mean
			case 2: reduce = true; break; // sum
		}
		dlprim::Shape target_shape;
		if(!reduce)
			target_shape = x.shape();
		Tensor loss_tensor = new_tensor_as(target_shape, self);
		dlprim::Tensor loss(todp(loss_tensor));
		if (reduce)
			dlprim::core::pointwiseOpBroadcastReduceStrided({x, y}, {loss}, {scale}, {},
				dlprim::core::PointwiseOp::eBcdFwdWeightless, dlprim::core::PointwiseOp::eAdd, {0.0});
		else
			dlprim::core::pointwiseOpBroadcastStrided({x, y}, {loss}, {scale},
				dlprim::core::PointwiseOp::eBcdFwdWeightless);
		return loss_tensor;
	}

	// {"schema": "aten::binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"} 
	Tensor & binary_cross_entropy_backward_out(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction, Tensor & grad_input)
	{
		GUARD;
		TORCH_CHECK(!weight || weight->numel()==0,"Weight in binar_cross_entroy isn't supported");
		dlprim::Tensor x = todp(self, true);
		dlprim::Tensor y = todp(target, true);
		dlprim::Tensor dloss = todp(grad_output, true);
		double scale = 1;
		if(reduction == 1) // mean
			scale = 1.0/x.shape().total_size(); 
		dlprim::Tensor dx = todp(grad_input, true);

		// -w (y - x) / (x - x^2)
		dlprim::core::pointwiseOpBroadcastStrided({x,y,dloss},{dx},{scale}, dlprim::core::PointwiseOp::eBceBwd);
		return grad_input;

	}

	// {"schema": "aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=Mean) -> Tensor", "dispatch": "True", "default": "False"}
	Tensor binary_cross_entropy_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, const c10::optional<Tensor> & weight, int64_t reduction)
	{
		GUARD;
		Tensor self_c = self.contiguous();
		Tensor input_grad = new_tensor_as(todp(self_c).shape(),self_c);
		binary_cross_entropy_backward_out(grad_output,self_c,target,weight,reduction,input_grad);
		return input_grad;
	}

	// {"schema": "aten::_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) grad_input) -> Tensor(a!)", "dispatch": "True", "default": "False"}
	Tensor & _softmax_backward_data_out(const Tensor & grad_output, const Tensor & output, int64_t dim, ScalarType /*input_dtype*/, Tensor & grad_input)
	{
#if 1
		return host_softmax_backward(
			SoftmaxEpilogue::eBackward,
			false,
			grad_output,
			output,
			dim,
			false,
			grad_input);
			
#else
		return impl_softmax_backward_data_out(grad_output,output,dim,false,grad_input);
#endif
	}

	
	// {"schema": "aten::mse_loss(Tensor self, Tensor target, int reduction=Mean) -> Tensor", "dispatch": "True", "default"
	Tensor mse_loss(const Tensor & self, const Tensor & target, int64_t reduction)
	{
		GUARD;
		dlprim::Tensor x = todp(self, true);
		dlprim::Tensor y = todp(target, true);
		float scale = 1.0;
		bool reduce = false;
		switch(reduction)
		{
			case 0: break;
			case 1: reduce = true; scale = scale = 1.0/x.shape().total_size(); break; // mean
			case 2: reduce = true; break; // sum
		}
		dlprim::Shape target_shape;
		if(!reduce)
			target_shape = x.shape();
		Tensor loss_tensor = new_tensor_as(target_shape, self);
		dlprim::Tensor loss(todp(loss_tensor));
		if (reduce)
			dlprim::core::pointwiseOpBroadcastReduceStrided({x, y}, {loss}, {scale}, {},
				dlprim::core::PointwiseOp::eMse, dlprim::core::PointwiseOp::eAdd, {0.0});
		else
			dlprim::core::pointwiseOpBroadcastStrided({x, y}, {loss}, {scale},
				dlprim::core::PointwiseOp::eMse);
		return loss_tensor;
	}
	// {"schema": "aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction)
	Tensor mse_loss_backward(const Tensor & grad_output, const Tensor & self, const Tensor & target, int64_t reduction)
	{
		GUARD;
		dlprim::Tensor x = todp(self, true);
		dlprim::Tensor dy = todp(grad_output, true);
		dlprim::Tensor lbl = todp(target, true);
		Tensor result = new_tensor_as(x.shape(),self);
		dlprim::Tensor dx = todp(result, true);
		double scale = reduction == 1 ? (1.0f/x.shape().total_size()) : 1.0;
		dlprim::core::pointwiseOpBroadcastStrided({dy, x, lbl}, {dx}, {scale}, dlprim::core::PointwiseOp::eMseBwd);
		return result;
	}


} // namespace dlprim
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
	  m.impl("aten::nll_loss_forward.output",&ptdlprim::nll_loss_forward_out);
	  m.impl("aten::nll_loss_backward.grad_input",&ptdlprim::nll_loss_backward_out);
	  m.impl("aten::binary_cross_entropy",&ptdlprim::binary_cross_entropy);
	  m.impl("aten::binary_cross_entropy_backward",&ptdlprim::binary_cross_entropy_backward);
	  m.impl("aten::binary_cross_entropy_backward.grad_input",&ptdlprim::binary_cross_entropy_backward_out);
	  m.impl("aten::mse_loss",&ptdlprim::mse_loss);
	  m.impl("aten::mse_loss_backward",&ptdlprim::mse_loss_backward);
}
