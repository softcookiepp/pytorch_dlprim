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

Tensor host_softmax(
	SoftmaxEpilogue epilogue,
	SoftmaxEpilogue epilogueWithMul,
	bool is_log_softmax,
	bool use_fast_softmax,
	const Tensor & input_,
	const int64_t dim_,
	const bool half_to_float,
	const Tensor& output)
{
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


}
