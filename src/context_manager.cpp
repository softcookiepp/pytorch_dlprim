// because having everything in a header is completely unreadable and slows down the build time of everything

namespace ptdlprim
{


static std::set<tart::buffer_ptr> CLContextManager::sAllocations;

CLContextManager& CLContextManager::instance()
{
	static std::once_flag once;
	static std::unique_ptr<CLContextManager> inst;
	std::call_once(once,init,inst);
	return *inst;
}

CLContextManager::~CLContextManager()
{
	{
		for(auto &data:data_)
			data->cache.clear();
	}
	no_cache_ = true;
}

unsigned CLContextManager::count()
{
	return instance().data_.size();
}
dlprim::Context CLContextManager::getContext(int id)
{
	return instance().data(id).ctx;
}

dlprim::ExecutionContext CLContextManager::getCommandQueue(int id)
{
	return instance().data(id).queue;
}

std::unique_ptr<CLMemAllocation> CLContextManager::alloc(int id,int64_t size)
{
	auto &d = instance().data(id);
	tart::device_ptr device = d.ctx.device();
	return d.cache.allocate(id, device, size);
}

// static
void CLContextManager::release(std::unique_ptr<CLMemAllocation> &&mem)
{
	auto &inst = instance();
	if(inst.no_cache_) {
		mem.reset();
		return;
	}
	auto &d = instance().data(mem->device_id);
	d.cache.release(std::move(mem));
}

// static
at::DataPtr CLContextManager::allocate(c10::Device const &dev,size_t n)
{
	#if 1
		auto& d = instance().data(dev.index());
		tart::device_ptr device = d.ctx.device();
		std::unique_ptr<CLMemAllocation> ptr = d.cache.allocate(dev.index(), device, n);
	#else
		std::unique_ptr<CLMemAllocation> ptr=alloc(dev.index(),n);
	#endif
	tart::buffer_ptr* buffer = &(ptr->buffer);
	return at::DataPtr(buffer,ptr.release(),&CLContextManager::free_ptr,dev);
}

// static
void CLContextManager::sync_if_needed(int index)
{
	auto &inst = instance();
	if(inst.no_cache_) {
		inst.data(index).queue.finish();
	}
}

//static
void CLContextManager::free_ptr(void *ctx)
{
	#if 0
		// TODO: write more concise stuff
	#else
		if(ctx == nullptr)
			return;
		std::unique_ptr<CLMemAllocation> ptr(static_cast<CLMemAllocation *>(ctx));
		release(std::move(ptr));
	#endif
}

//static
dlprim::RandomState& CLContextManager::rng_state(int index)
{
	return instance().data_.at(index)->rng;
}

//static
bool CLContextManager::is_ready(int index)
{
	auto &data = instance().data_;
	if(index < 0 || index >= int(data.size()) || !data[index])
		return false;
	return data[index]->ready;
}

//static
bool CLContextManager::fp64(int index)
{
	#if 0
	#else
		return instance().data(index).fp64;
	#endif
}

//static
bool CLContextManager::enable_profiling(int device)
{
	if(is_ready(device))
		return false;
	if(unsigned(device) >= count())
		return false;
	instance().data_.at(device)->enable_profiling = true;
	return true;
}

//static
void CLContextManager::clear(int index)
{
	auto &data = instance().data_;
	if(index < 0 || index >= int(data.size()) || !data[index] || !data[index]->ready)
		return;
	getCommandQueue(index).finish();
	data[index]->cache.clear();
}

//static
bool CLContextManager::bad_fork()
{
	instance();
	return bad_fork_;
}

#if 0
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
#endif


//static
void CLContextManager::init(std::unique_ptr<CLContextManager> &self)
{
	self.reset(new CLContextManager());
	self->allocate();
}

// Called in the forked child if cuda has already been initialized
//static
void CLContextManager::forked_child()
{
	bad_fork_ = true;
}

//static
void CLContextManager::poison_fork() {
	static c10::once_flag flag;
	c10::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
}

void CLContextManager::allocate()
{
	poison_fork();
	char *no_cache=getenv("OPENCL_NO_MEM_CACHE");
	no_cache_ = no_cache && atoi(no_cache);
	
	tart::Instance& tartInstance = dlprim::Context::getInstance();
	for(size_t j=0; j < tartInstance.getNumDevices(); j++) {
		std::unique_ptr<DevData> d(new DevData());
		data_.push_back(std::move(d));
		data_.back()->name = "0:" + std::to_string(j);
	}
}

DevData& CLContextManager::data(int i)
{
	if(i < 0)
		i = 0;
	if(i >= int(data_.size()))
		throw std::runtime_error("Invalid Device #" + std::to_string(i));
	DevData &res = *data_[i];
	if(res.ready)
		return res;
	std::cout << "	res.name: " << res.name << std::endl;
	res.ctx = dlprim::Context(res.name);
	res.fp64 = res.ctx.device()->getMetadata().double_;
	res.queue = res.ctx.make_execution_context(0);
	res.cache.prepare(res.ctx);
	res.ready = true;
	std::cout << "Accessing device #" << i << ":" << res.ctx.name() << std::endl;
	return res;
}


} // namespace ptdlprim
