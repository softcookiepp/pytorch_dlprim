// this is what needs to be ported
template <int nt, int vt, typename func_t>
C10_LAUNCH_BOUNDS_2(nt, 4)
__global__ void elementwise_kernel(int N, func_t f) {
	int tid = threadIdx.x;
	int nv = nt * vt;
	int idx = nv * blockIdx.x + tid;
#pragma unroll
	for (int i = 0; i < vt; i++) {
		if (idx < N) {
			f(idx);
			idx += nt;
		}
	}
}

template <int nt, int vt, typename func_t>
static void launch_legacy_kernel(int64_t N, const func_t& f) {
	TORCH_INTERNAL_ASSERT(N >= 0 && N <= std::numeric_limits<int32_t>::max());
	if (N == 0) {
		return;
	}
	dim3 block(nt);
	dim3 grid((N + block.x * vt - 1) / (block.x * vt));
	auto stream = at::cuda::getCurrentCUDAStream();
	elementwise_kernel<nt, vt, func_t><<<grid, block, 0, stream>>>(N, f);
	C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename func_t>
void gpu_kernel_impl(TensorIteratorBase& iter, const func_t& f) {
	if (!needs_dynamic_casting<func_t>::check(iter)) {
		return gpu_kernel_impl_nocast(iter, f);
	}
	using traits = function_traits<func_t>;
	using arg0_t = typename traits::result_type;
	constexpr int ntensors = traits::arity + 1; // input arguments + one output tensor

	TORCH_INTERNAL_ASSERT(iter.can_use_32bit_indexing());
	TORCH_INTERNAL_ASSERT(iter.ninputs() == traits::arity);
	TORCH_INTERNAL_ASSERT(iter.noutputs() == 1);

	std::array<char*, ntensors> data;
	for (int i = 0; i < ntensors; i++) {
		data[i] = (char*)iter.data_ptr(i);
	}

	int64_t numel = iter.numel();

	bool contiguous = iter.is_contiguous();

	if (contiguous)
	{
		auto loader = memory::LoadWithCast<traits::arity>(iter);
		auto storer = memory::StoreWithCast<1>(iter);
		auto input_offset_calculator = TrivialOffsetCalculator<traits::arity>();
		auto output_offset_calculator = TrivialOffsetCalculator<1>();
		launch_unrolled_kernel(
				numel,
				f,
				data,
				input_offset_calculator,
				output_offset_calculator,
				loader,
				storer);
	}
	else
	{
		std::array<ScalarType, ntensors> dtypes;
		for (int i = 0; i < ntensors; i++) {
			dtypes[i] = iter.dtype(i);
		}
		auto offset_calc = ::make_offset_calculator<traits::arity + 1>(iter);

		launch_legacy_kernel<128, 4>(numel, [=] GPU_LAMBDA(int idx) {
			auto offsets = offset_calc.get(idx);
			void* out = data[0] + offsets[0];
			arg0_t result = invoke(f, &data[1], &offsets[1], &dtypes[1], 1);
			c10::cast_and_store<arg0_t>(dtypes[0], out, result);
		});
	}
}

template <typename func_t>
void gpu_kernel(TensorIteratorBase& iter, const func_t& f) {

#if 0 // check for Vulkan later
	for (int arg = 0; arg < iter.ntensors(); arg++) {
		TORCH_CHECK(
			iter.device(arg).is_cuda(),
			"argument ", arg, ": expected a CUDA device but found ", iter.device(arg));
	}
#endif
	if (iter.numel() == 0) {
		return;
	}
	
	// this may cause other problems later. though I am not certain yet
	if (!iter.can_use_32bit_indexing()) {
		for (auto& sub_iter : iter.with_32bit_indexing()) {
			gpu_kernel(sub_iter, f);
		}
		return;
	}

	gpu_kernel_impl(iter, f);
}
