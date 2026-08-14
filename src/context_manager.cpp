// because having everything in a header is completely unreadable and slows down the build time of everything
#include "CLTensor.h"

namespace ptdlprim
{


std::set<tart::buffer_ptr> CLContextManager::sAllocations;
std::map<int, CLCache> CLContextManager::sDeviceCaches;

CLContextManager& CLContextManager::instance()
{
	static std::once_flag once;
	static std::unique_ptr<CLContextManager> inst;
	std::call_once(once,init,inst);
	return *inst;
}

CLContextManager::~CLContextManager()
{
	// wait, do we really need this?
	for (auto& pair : sDeviceCaches)
		pair.second.clear();
	no_cache_ = true;
}

//static
CLCache& CLContextManager::getCache(int deviceIndex)
{
	if (sDeviceCaches.find(deviceIndex) == sDeviceCaches.end())
	{
		tart::device_ptr device = dlprim::Context::getInstance().getDevice(deviceIndex);
		sDeviceCaches[deviceIndex].prepare(device);
	}
	return sDeviceCaches[deviceIndex];
}

//static
unsigned CLContextManager::count()
{
	return dlprim::Context::getInstance().getNumDevices();
}

//static
dlprim::Context CLContextManager::getContext(int id)
{
	tart::device_ptr device = dlprim::Context::getInstance().getDevice(id);
	return dlprim::Context(device);
}

// static
void CLContextManager::release(std::unique_ptr<CLMemAllocation> &&mem)
{
	auto &inst = instance();
	if(inst.no_cache_) {
		mem.reset();
		return;
	}
	getCache(mem->device_id).release(std::move(mem));
}

// static
at::DataPtr CLContextManager::allocate(c10::Device const &dev,size_t n)
{
	tart::device_ptr device = dlprim::Context::getInstance().getDevice(dev.index());
	std::unique_ptr<CLMemAllocation> ptr = getCache(dev.index()).allocate(dev.index(), device, n);
	tart::buffer_ptr* buffer = &(ptr->buffer);
	return at::DataPtr(buffer,ptr.release(),&CLContextManager::free_ptr,dev);
}

// static
void CLContextManager::sync_if_needed(int index)
{
	auto &inst = instance();
	if(inst.no_cache_)
	{
		dlprim::Context::getInstance().getDevice(index)->sync();
	}
}

//static
void CLContextManager::free_ptr(void *ctx)
{
	if(ctx == nullptr)
		return;
	std::unique_ptr<CLMemAllocation> ptr(static_cast<CLMemAllocation *>(ctx));
	release(std::move(ptr));
}

//static
dlprim::RandomState& CLContextManager::rng_state(int index)
{
	static std::map<int, dlprim::RandomState> rngs;
	return rngs[index];
}

//static
bool CLContextManager::is_ready(int index)
{
	getCache(index);
	return true;
}

//static
bool CLContextManager::enable_profiling(int device)
{
	return false;
}

//static
void CLContextManager::clear(int index)
{
	dlprim::Context::getInstance().getDevice(index)->sync();
	sDeviceCaches[index].clear();
}

//static
bool CLContextManager::bad_fork()
{
	instance();
	return bad_fork_;
}

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
}


} // namespace ptdlprim
