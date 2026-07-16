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
	Tensor& output)
{
	std::cout << "WEEEEEEEEEEEEEEEEEE" << std::endl;
	if (half_to_float)
	{
		throw std::runtime_error("not implemented");
		TORCH_CHECK(input_.scalar_type() == ScalarType::Half, "conversion is supported for Half type only");
	}
	auto input = input_.contiguous();
	//static_assert(std::is_same_v<acc_type<at::Half, true>, float>, "accscalar_t for half should be float");
	if (input.dim() == 0) input = input.view(1);
	int64_t dim = maybe_wrap_dim(dim_, input.dim());
	TORCH_CHECK(dim >=0 && dim < input.dim(), "dim must be non-negative and less than input dimensions");
	int64_t outer_size = 1;
	int64_t dim_size = input.size(dim);

	if (input.numel() > 0)
	{
		int64_t inner_size = 1;
		auto stream = getExecutionContext(input_);
		for (int64_t i = 0; i < dim; ++i)
			outer_size *= input.size(i);
		for (int64_t i = dim + 1; i < input.dim(); ++i)
			inner_size *= input.size(i);
		// This kernel spawns a block per each element in the batch.
		// XXX: it assumes that inner_size == 1
#if 0 // TODO: implement (possibly some) of this stuff
		if (inner_size == 1)
		{
			dim3 grid(outer_size);
			AT_DISPATCH_FLOATING_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, input.scalar_type(), "host_softmax", [&] {
				using accscalar_t = acc_type<scalar_t, true>;
				if (!half_to_float) {
					auto output_ptr = output.mutable_data_ptr<scalar_t>();
					auto input_ptr = input.const_data_ptr<scalar_t>();
					if (dim_size <= 2048 && dim_size*sizeof(scalar_t) <= 8192) {
						int64_t remaining = outer_size;
						int64_t chunk_size = (1L << 30L) / dim_size;
						while(remaining > 0) {
							dispatch_softmax_forward<scalar_t, scalar_t, accscalar_t, is_log_softmax, false>(
								output_ptr, input_ptr, dim_size, dim_size, std::min<int64_t>(remaining, chunk_size), nullptr/* not masked */);
							input_ptr += chunk_size * dim_size;
							output_ptr += chunk_size * dim_size;
							remaining -= chunk_size;
						}
					} else {
						constexpr int ILP = sizeof(float4) / sizeof(scalar_t);
						if constexpr (use_fast_softmax) {
							dim3 block(512);
							size_t smem_reduction_sz = block.x / at::cuda::warp_size() * sizeof(accscalar_t);
							if (dim_size % ILP == 0) {
								cunn_SoftMaxForwardGmem<ILP, scalar_t, accscalar_t, scalar_t, EpilogueWithMul>
										<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
							} else {
								cunn_SoftMaxForwardFast<ILP, scalar_t, accscalar_t, scalar_t, EpilogueWithMul>
										<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
							}
						} else {
							dim3 block = SoftMaxForward_getBlockSize(dim_size);
							size_t smem_reduction_sz = block.x / at::cuda::warp_size() * sizeof(accscalar_t);
							auto max_elements_per_smem = (at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock -
								smem_reduction_sz) / sizeof(scalar_t);

							bool can_use_smem = static_cast<size_t>(dim_size) < max_elements_per_smem;
							can_use_smem &= !(reinterpret_cast<uintptr_t>(input_ptr) % ALIGN_BYTES);
							can_use_smem &= (!(reinterpret_cast<uintptr_t>(output_ptr) % ALIGN_BYTES));
							can_use_smem &= !(dim_size % ILP);

							int32_t potential_reg_cnt = potential_register_count(dim_size, block.x);
							if(potential_reg_cnt < 10){
								TORCH_INTERNAL_ASSERT(potential_reg_cnt > 0, "potential_reg_cnt for softmax with register should be greater than 0.");
								switch (potential_reg_cnt) {
									// TODO(Wenqin): try to investigate why we couldn't use macro for below code,
									// because it seems on MSVS, it seems the macro way didn't expand correct.
									case 1:
										cunn_SoftMaxForwardReg<scalar_t, accscalar_t, scalar_t, Epilogue, int64_t, 1>
											<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
										break;
									case 2:
										cunn_SoftMaxForwardReg<scalar_t, accscalar_t, scalar_t, Epilogue, int64_t, 2>
											<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
										break;
									case 3:
										cunn_SoftMaxForwardReg<scalar_t, accscalar_t, scalar_t, Epilogue, int64_t, 3>
											<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
										break;
									case 4:
										cunn_SoftMaxForwardReg<scalar_t, accscalar_t, scalar_t, Epilogue, int64_t, 4>
											<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
										break;
									case 5:
										cunn_SoftMaxForwardReg<scalar_t, accscalar_t, scalar_t, Epilogue, int64_t, 5>
											<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
										break;
									case 6:
										cunn_SoftMaxForwardReg<scalar_t, accscalar_t, scalar_t, Epilogue, int64_t, 6>
											<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
										break;
									case 7:
										cunn_SoftMaxForwardReg<scalar_t, accscalar_t, scalar_t, Epilogue, int64_t, 7>
											<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
										break;
									case 8:
										cunn_SoftMaxForwardReg<scalar_t, accscalar_t, scalar_t, Epilogue, int64_t, 8>
											<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
										break;
									case 9:
										cunn_SoftMaxForwardReg<scalar_t, accscalar_t, scalar_t, Epilogue, int64_t, 9>
											<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
										break;
								}
							} else if (can_use_smem) {
								size_t smem_sz = dim_size * sizeof(scalar_t) + smem_reduction_sz;
								cunn_SoftMaxForwardSmem<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
									<<<grid, block, smem_sz, stream>>>(output_ptr, input_ptr, dim_size);
							} else {
								cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, scalar_t, Epilogue>
									<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
							}
						}

						C10_CUDA_KERNEL_LAUNCH_CHECK();
					}
				} else {
					auto output_ptr = output.mutable_data_ptr<accscalar_t>();
					auto input_ptr = input.const_data_ptr<scalar_t>();
					if (dim_size <= 1024 && dim_size*sizeof(scalar_t) <= 4096) {
						int64_t remaining = outer_size;
						int64_t chunk_size = (1<<30) / dim_size;
						while(remaining > 0) {
							dispatch_softmax_forward<scalar_t, accscalar_t, accscalar_t, is_log_softmax, false>(
									output_ptr, input_ptr, dim_size, dim_size, std::min<int64_t>(remaining, chunk_size), nullptr/* not masked */);
							input_ptr += chunk_size * dim_size;
							output_ptr += chunk_size * dim_size;
							remaining -= chunk_size;
						}
					} else {
						constexpr int ILP = sizeof(float4) / sizeof(scalar_t);
						if constexpr (use_fast_softmax) {
							dim3 block(512);
							size_t smem_reduction_sz = block.x / at::cuda::warp_size() * sizeof(accscalar_t);
							if (dim_size % ILP == 0) {
								cunn_SoftMaxForwardGmem<ILP, scalar_t, accscalar_t, accscalar_t, EpilogueWithMul>
										<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
							} else {
								cunn_SoftMaxForwardFast<ILP, scalar_t, accscalar_t, accscalar_t, EpilogueWithMul>
										<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
							}
						} else {
							dim3 block = SoftMaxForward_getBlockSize(dim_size);
							size_t smem_reduction_sz = block.x / at::cuda::warp_size() * sizeof(accscalar_t);
							auto max_elements_per_smem = (at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock -
								smem_reduction_sz) / sizeof(scalar_t);

							bool can_use_smem = static_cast<size_t>(dim_size) < max_elements_per_smem;
							can_use_smem &= !(reinterpret_cast<uintptr_t>(input_ptr) % ALIGN_BYTES);
							can_use_smem &= (!(reinterpret_cast<uintptr_t>(output_ptr) % ALIGN_BYTES));
							can_use_smem &= !(dim_size % ILP);

							if (can_use_smem) {
								size_t smem_sz = dim_size * sizeof(scalar_t) + smem_reduction_sz;
								cunn_SoftMaxForwardSmem<ILP, scalar_t, accscalar_t, accscalar_t, Epilogue>
									<<<grid, block, smem_sz, stream>>>(output_ptr, input_ptr, dim_size);
							} else {
								cunn_SoftMaxForward<ILP, scalar_t, accscalar_t, accscalar_t, Epilogue>
									<<<grid, block, smem_reduction_sz, stream>>>(output_ptr, input_ptr, dim_size);
							}
						}

						C10_CUDA_KERNEL_LAUNCH_CHECK();
					}
				}
			});
		// This kernel runs in a 2D grid, where each application along y dimension has a fixed
		// outer_size, and runs in parallel over inner_size. Dimension x is parallel over outer_size.
		// Reductions over dim are done in a single-threaded manner.
		}
		else
#endif
		{
			auto output_dp = todp(output);
			auto input_dp = todp(input_);
			
			if (output.dtype() != input_.dtype())
				throw std::invalid_argument("input and output for softmax must be same dtype");
			
			if (epilogue != SoftmaxEpilogue::eForward)
				throw std::runtime_error("only forward, non-log softmax is implemented");
			
			dlprim::gpu::spatial_softmax(
				stream,
				todp(output.dtype()),
				epilogue,
				output_dp.device_buffer(),
				output_dp.device_offset(),
				input_dp.device_buffer(),
				input_dp.device_offset(),
				outer_size,
				dim_size,
				inner_size,
				half_to_float);
		}
	}
	return output;
}

#if 0
template<typename input_t, typename output_t, typename accscalar_t, template<typename, typename, typename> class Epilogue>
void dispatch_host_softmax_backward(int64_t dim_size, dim3 grid, Tensor &grad, Tensor &output, const Tensor &gI)
{
	cudaStream_t stream = at::cuda::getCurrentCUDAStream();
	constexpr int ILP = sizeof(float4) / sizeof(output_t);
	dim3 block = SoftMax_getBlockSize(ILP, dim_size);

	size_t smem_reduction_sz = block.x / at::cuda::warp_size() * sizeof(accscalar_t);
	auto max_elements_per_smem = (at::cuda::getCurrentDeviceProperties()->sharedMemPerBlock -
		smem_reduction_sz) / sizeof(output_t);
	bool can_use_smem = static_cast<size_t>(dim_size) < max_elements_per_smem;
	can_use_smem &= (!(reinterpret_cast<uintptr_t>(gI.const_data_ptr<input_t>()) % ALIGN_BYTES));
	can_use_smem &= (!(reinterpret_cast<uintptr_t>(output.const_data_ptr<output_t>()) % ALIGN_BYTES));
	can_use_smem &= !(reinterpret_cast<uintptr_t>(grad.const_data_ptr<output_t>()) % ALIGN_BYTES);
	can_use_smem &= !(dim_size % ILP);
	// This should not be needed on current generation GPUs because the size of shared memory is so low.
	// But we add this check to be defensive and future-proof just in case shared memory size goes up
	// to be so large as to requires 64-bits of addressing.
	can_use_smem &= (dim_size < std::numeric_limits<int32_t>::max());

	if (can_use_smem) {
		size_t smem_sz = dim_size * sizeof(output_t) + smem_reduction_sz;
		cunn_SoftMaxBackwardSmem<ILP, input_t, accscalar_t, output_t, Epilogue>
		<<<grid, block, smem_sz, stream>>>(
			gI.mutable_data_ptr<input_t>(), output.const_data_ptr<output_t>(), grad.const_data_ptr<output_t>(), dim_size);
	} else {
		cunn_SoftMaxBackward<ILP, input_t, accscalar_t, output_t, Epilogue>
		<<<grid, block, block.x * sizeof(accscalar_t), stream>>>(
				gI.mutable_data_ptr<input_t>(), output.const_data_ptr<output_t>(), grad.const_data_ptr<output_t>(), dim_size
			);
	}
	C10_CUDA_KERNEL_LAUNCH_CHECK();
}
#endif

Tensor& host_softmax_backward(
	SoftmaxEpilogue epilogue,
	bool is_log_softmax,
	const Tensor &grad_,
	const Tensor &output_,
	int64_t dim_,
	bool half_to_float,
	Tensor &gI)
{
	int64_t dim = maybe_wrap_dim(dim_, grad_.dim());
	if (grad_.numel() == 0)
	{
		return gI;
	}
	auto grad = grad_.contiguous();
	
	if (grad.dim() == 0) grad = grad.view(1);
	TORCH_CHECK(dim >=0 && dim < grad.dim(), "dim must be non-negative and less than input dimensions");
	auto output = output_.contiguous();
	if (output.dim() == 0) output = output.view(1);
	int64_t outer_size = 1;
	int64_t dim_size = output.size(dim);
	int64_t inner_size = 1;
	for (int64_t i = 0; i < dim; ++i)
		outer_size *= output.size(i);
	for (int64_t i = dim + 1; i < output.dim(); ++i)
		inner_size *= output.size(i);
		
// See descriptions of kernels above.
	auto stream = getExecutionContext(output_);
	if (inner_size == 1)
	{
#if 1
		throw std::runtime_error("not implemented");
#else
		dim3 grid(outer_size);
		using accscalar_t = acc_type<scalar_t, true>;
		if (!half_to_float)
		{
			if (dim_size <= 1024 && dim_size*sizeof(scalar_t) <= 4096) {
				auto gI_ptr = gI.mutable_data_ptr<scalar_t>();
				auto grad_ptr = grad.const_data_ptr<scalar_t>();
				auto output_ptr = output.const_data_ptr<scalar_t>();
				int64_t remaining = outer_size;
				int64_t chunk_size = (1<<30) / dim_size;
				while(remaining > 0) {
					dispatch_softmax_backward<scalar_t, scalar_t, accscalar_t, is_log_softmax, false /* masked_softmax */>(
						gI_ptr, grad_ptr, output_ptr, dim_size, dim_size, std::min<int64_t>(remaining, chunk_size));
					gI_ptr += chunk_size * dim_size;
					grad_ptr += chunk_size * dim_size;
					output_ptr += chunk_size * dim_size;
					remaining -= chunk_size;
				}
			} else {
				dispatch_host_softmax_backward<scalar_t, scalar_t, accscalar_t, Epilogue>(dim_size, grid, grad, output, gI);
			}
		}
		else
		{
			if (dim_size <= 1024 && dim_size*sizeof(scalar_t) <= 4096)
			{
				auto gI_ptr = gI.mutable_data_ptr<scalar_t>();
				auto grad_ptr = grad.const_data_ptr<accscalar_t>();
				auto output_ptr = output.const_data_ptr<accscalar_t>();
				int64_t remaining = outer_size;
				int64_t chunk_size = (1<<30) / dim_size;
				while(remaining > 0) {
					dispatch_softmax_backward<accscalar_t, scalar_t, accscalar_t, is_log_softmax, false /* masked_softmax */>(
						gI_ptr, grad_ptr, output_ptr, dim_size, dim_size, std::min<int64_t>(remaining, chunk_size));
					gI_ptr += chunk_size * dim_size;
					grad_ptr += chunk_size * dim_size;
					output_ptr += chunk_size * dim_size;
					remaining -= chunk_size;
				}
			}
			else
			{
				dispatch_host_softmax_backward<scalar_t, accscalar_t, accscalar_t, Epilogue>(dim_size, grid, grad, output, gI);
			}
		}
#endif
	}
	else
	{
		std::cout << "DOING SOFTMAX BACKWARD" << std::endl;
		// this goes in dlprimitives::gpu
		auto gI_dp = todp(gI);
		auto output_dp = todp(output);
		auto grad_dp = todp(grad);
		
		dlprim::gpu::spatial_softmax_backward(
			stream,
			todp(output.dtype()),
			epilogue,
			gI_dp.device_buffer(),
			gI_dp.device_offset(),
			output_dp.device_buffer(),
			output_dp.device_offset(),
			grad_dp.device_buffer(),
			grad_dp.device_offset(),
			outer_size,
			dim_size,
			inner_size,
			false);
	}
	return gI;
}

}
