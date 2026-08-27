#include "utils.h"

namespace ptdlprim
{
	void sync_if_needed(c10::Device const &d)
	{
        CLContextManager::sync_if_needed(d.index());
    }

    const tart::DType& todp(c10::ScalarType tp);
    
    const tart::DType& todp(caffe2::TypeMeta meta)
    {
        return todp(meta.toScalarType());
    }

    const tart::DType& todp(c10::ScalarType tp)
    {
        switch(tp)
        {
			case c10::kFloat:
				return tart::dtypes::float32;
			case c10::kDouble:
				tart::dtypes::float64;
			case c10::kHalf:
				tart::dtypes::float16;
			#if 0
				// not implemented yet, sorries
				case c10::kBFloat16:
					return dlprim::bfloat16_data;
			#endif
			case c10::kLong:
				return tart::dtypes::int64;
			case c10::kInt:
				return tart::dtypes::int32;
			case c10::kShort:
				return tart::dtypes::int16;
			case c10::kChar:
				return tart::dtypes::int8;
			case c10::kByte:
				return tart::dtypes::uint8;
			case c10::kBool:
				TORCH_CHECK(sizeof(bool)==1,"Need to make sure tensors have same size");
				return tart::dtypes::uint8;
			default:
				throw std::runtime_error(std::string("Unsupported data type:") + c10::toString(tp));
        }
    }

	tart::buffer_ptr buffer_from_tensor(torch::Tensor const &tt)
    {
        TORCH_CHECK(tt.device().type() == OpenCLDeviceType,"OpenCL device is required for tensor");
        //TORCH_CHECK(tt.numel() > 0,"Buffer is not valid for unallocated defvice");
        TORCH_CHECK(tt.getIntrusivePtr()->storage().nbytes() > 0,"Buffer is not valid for unallocated defvice");
        
        // pretty sure this will work
        tart::buffer_ptr* p = static_cast<tart::buffer_ptr*>(const_cast<void*>(tt.getIntrusivePtr()->storage().data()));
        return *p;
    }

    dlprim::Tensor todp(torch::Tensor const &tt, const bool skipContiguousCheck)
    {
        TORCH_CHECK(tt.device().type() == OpenCLDeviceType,"OpenCL device is required for tensor");
        // So this is going to cause problems, and may explain why this backend is failing to pass a lot of the tests copied directly from torch.
        if (!skipContiguousCheck)
			TORCH_CHECK(tt.is_contiguous(),"dlprim::Tensor must be contiguous");
        auto sizes = tt.sizes();
        auto strides = tt.strides();
        auto offset = tt.storage_offset();
        auto dtype = tt.dtype();
        tart::buffer_ptr buf = buffer_from_tensor(tt);
        dlprim::Shape sp;
        dlprim::Shape st;
        if(sizes.empty())
		{
			// This may be causing some problems.
            sp = dlprim::Shape(1); // scalar
            st = dlprim::Shape(1);
		}
        else
        {
            sp = dlprim::Shape::from_range(sizes.begin(), sizes.end());
            st = dlprim::Shape::from_range(strides.begin(), strides.end());
		}
        dlprim::Tensor res(buf,offset,sp, st, todp(dtype));
        if (!res.isContiguous() && !skipContiguousCheck)
		{
			std::cout << "	torch shape: ";
			for (auto& s : tt.sizes())
				std::cout << s << ", ";
			std::cout << std::endl;
			std::cout << "	torch strides: ";
			for (auto& s : tt.strides())
				std::cout << s << ", ";
			std::cout << std::endl;
			std::cout << "	dlprim shape: ";
			for (auto s : res.shape())
				std::cout << s << ", ";
			std::cout << std::endl;
			std::cout << "	dlprim strides: ";
			for (auto s : res.stride())
				std::cout << s << ", ";
			std::cout << std::endl;
			if (tt.is_contiguous())
				throw std::runtime_error("dlprim tensor marked as non-contiguous, when it should be marked as contiguous");
			throw std::runtime_error("accidentally made non-contiguous tensor");
		}
        return res;
    }

    torch::Tensor new_ocl_tensor(torch::IntArrayRef size,c10::Device dev,c10::ScalarType type)
    {
        size_t n = 1;
        for(auto const &v:size)
            n*=v;
        tart::DType dt = todp(type);
        size_t mem = std::max(size_t(1),n)*dt.size();
        c10::Storage storage(c10::Storage::use_byte_size_t(),mem,CLContextManager::allocate(dev,mem));

        c10::DispatchKeySet ks = c10::DispatchKeySet{c10::DispatchKey::OpenCL, c10::DispatchKey::AutogradOpenCL};
        
        c10::intrusive_ptr<c10::TensorImpl> impl=c10::make_intrusive<c10::TensorImpl>(
            std::move(storage),
            ks,
            caffe2::TypeMeta::fromScalarType(type));

        impl->set_sizes_contiguous(size);


        return torch::Tensor::wrap_tensor_impl(impl);

    }

    torch::Tensor new_tensor_as(dlprim::Shape const &s,torch::Tensor const &as)
    {
        int64_t shape[dlprim::max_tensor_dim];
        for(int i=0;i<s.size();i++)
            shape[i]=s[i];
        torch::Tensor result = new_ocl_tensor(c10::IntArrayRef(shape,s.size()),
                                              as.device(),
                                              as.dtype().toScalarType());
        return result;
    }

    dlprim::Tensor make_workspace(at::DataPtr &ws_ptr,size_t ws_size,c10::Device const &dev)
    {
        dlprim::Tensor ws;
        if(ws_size) {
            ws_ptr = std::move(CLContextManager::allocate(dev,ws_size));
            ws = dlprim::Tensor( *((tart::buffer_ptr*)ws_ptr.get()), 0, dlprim::Shape(ws_size), tart::dtypes::uint8);
        }
        return ws;
    }


} // namespace
