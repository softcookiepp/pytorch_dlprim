#pragma once
#include <dlprim/core/common.hpp>
#include <dlprim/random.hpp>

#define OpenCL PrivateUse1
#define AutogradOpenCL AutogradPrivateUse1
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <c10/core/Storage.h>

#include <mutex>
#include <cstdint>
#include <list>
#include <set>

#define PTD_TIMER_GUARD(function_name) tart::TimerGuard _TG_(function_name, gProfiler)

#define ASSERT_DLPRIM(tensor_) { if () }

namespace ptdlprim {
	
extern tart::profiler_ptr gProfiler;

#ifdef USE_PATCHED_TORCH
    constexpr c10::DeviceType OpenCLDeviceType = c10::DeviceType::OPENCL;
#else
    constexpr c10::DeviceType OpenCLDeviceType = c10::DeviceType::PrivateUse1;
#endif    


    class ExecGuard {
    public:
        static void set_profiling_context(dlprim::ExecutionContext *queue = nullptr);
        ExecGuard(char const *name,char const *short_name);
        ~ExecGuard();
    private:
        char const *name_;
    };
    
	#ifdef _MSC_VER 
	#  define GUARD ExecGuard debug_guard(__FUNCSIG__,__func__);
	#else
    #  define GUARD ExecGuard debug_guard(__PRETTY_FUNCTION__,__func__);
	#endif

    struct CLMemAllocation {

        CLMemAllocation(CLMemAllocation const &) = delete;
        void operator=(CLMemAllocation const &) = delete;
        CLMemAllocation(CLMemAllocation &&) = default;
        CLMemAllocation &operator=(CLMemAllocation &&) = default;
        ~CLMemAllocation() {}
		CLMemAllocation(int id, tart::device_ptr& ctx, std::int64_t length, std::int64_t os) :
            device_id(id),
            size(length),
            orig_size(os)
        {
			buffer = ctx->allocateBuffer(length);

        }
        int device_id;
        std::int64_t size;
        std::int64_t orig_size;
		tart::buffer_ptr buffer = nullptr;
    };

    class CLCache {
    public:
        CLCache() {}

        CLCache(CLCache const &) = delete;
        void operator=(CLCache const &) = delete;
        typedef std::map<std::int64_t,std::list<std::unique_ptr<CLMemAllocation> > > allocation_type;
        std::mutex lock;
        allocation_type allocation;

        std::int64_t allocated_size = 0;
        std::int64_t peak_requested_size = 0;
        std::int64_t requested_size = 0;
        std::int64_t cached_size = 0;

        bool reuse_oversized_chunks = getenv("OPENCL_CACHE_OVERSIZED") && atoi(getenv("OPENCL_CACHE_OVERSIZED"));
        bool debug_allocator = (getenv("OPENCL_DEBUG_CACHE") && atoi(getenv("OPENCL_DEBUG_CACHE")));

        void clear();
        static std::uint64_t round(uint64_t v);
#if VULKAN_API
		std::unique_ptr<CLMemAllocation> allocate(int id, tart::device_ptr& ctx, int64_t orig_size);
#else
        std::unique_ptr<CLMemAllocation> allocate(int id,cl::Context &ctx,int64_t orig_size);
#endif
        void release(std::unique_ptr<CLMemAllocation> &&mem);
        //void prepare(dlprim::Context &ctx);
        void prepare(const tart::device_ptr& device);
    };
    

    class CLContextManager
    {
		static std::set<tart::buffer_ptr> sAllocations;
		static std::map<int, CLCache> sDeviceCaches;
    public: 
        static CLContextManager &instance();
        ~CLContextManager();
        static unsigned count();
        static dlprim::Context getContext(int id);
        static dlprim::ExecutionContext getCommandQueue(int id);
        static std::unique_ptr<CLMemAllocation> alloc(int id,int64_t size);
        static void release(std::unique_ptr<CLMemAllocation> &&mem);
        static at::DataPtr allocate(c10::Device const &dev,size_t n);

        static void sync_if_needed(int index);

        static void free_ptr(void *ctx);

        static dlprim::RandomState &rng_state(int index);
        static bool is_ready(int index);

        static bool fp64(int index);

        static bool enable_profiling(int device);
        static void start_profiling(int device);
        static void stop_profiling(int device,std::string const &output);
        static void clear(int index);
        
        static bool bad_fork();
        
        static CLCache& getCache(int device);

    private:

        struct DevData {
            bool ready = false; // FIXME make thread safe
            bool enable_profiling = false;
            bool fp64 = false;
            dlprim::RandomState rng;
            std::string name;
            dlprim::Context ctx;
            dlprim::ExecutionContext queue;
            CLCache cache;
            std::shared_ptr<dlprim::TimingData> timing;
        };



        static void init(std::unique_ptr<CLContextManager> &self);

        // Called in the forked child if cuda has already been initialized
        static void forked_child();

        static void poison_fork();
        
        void allocate();

        DevData &data(int i);
        
        std::vector<std::unique_ptr<DevData> > data_;
        bool no_cache_;
        static bool bad_fork_;
    };




} // namespace ptdlprim
