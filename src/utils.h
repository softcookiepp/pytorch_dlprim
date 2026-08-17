#ifndef PTDLPRIM_UTILS_H
#define PTDLPRIM_UTILS_H

#include "CLTensor.h"

namespace ptdlprim {
    /// 
    void sync_if_needed(c10::Device const &d);

    const tart::DType& todp(c10::ScalarType tp);
    
    const tart::DType& todp(caffe2::TypeMeta meta);

	tart::buffer_ptr buffer_from_tensor(torch::Tensor const &tt);

    dlprim::Tensor todp(torch::Tensor const &tt);
    torch::Tensor new_ocl_tensor(torch::IntArrayRef size,c10::Device dev,c10::ScalarType type=c10::kFloat);

    torch::Tensor new_tensor_as(dlprim::Shape const &s,torch::Tensor const &as);
    dlprim::Tensor make_workspace(at::DataPtr &ws_ptr,size_t ws_size,c10::Device const &dev);

    class WSGuard {
    public:
        WSGuard(size_t size,c10::Device const &dev)
        {
            ws = make_workspace(ws_ptr_,size,dev);
        }
        dlprim::Tensor ws;
    private:
        at::DataPtr ws_ptr_;
    };



} // namespace

#endif // PTDLPRIM_UTILS_H
