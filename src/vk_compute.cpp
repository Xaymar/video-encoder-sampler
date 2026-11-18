#include "vk_compute.h"
#include "gpu.h"

#ifndef VES_NO_VULKAN
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "shaders_spv.h"

#define VK_FUNCS(X) \
	X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties) X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceQueueFamilyProperties) \
	X(vkGetPhysicalDeviceFeatures) X(vkCreateDevice) X(vkGetDeviceQueue) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) \
	X(vkAllocateMemory) X(vkFreeMemory) X(vkBindBufferMemory) X(vkMapMemory) X(vkCreateShaderModule) X(vkCreateDescriptorSetLayout) \
	X(vkCreatePipelineLayout) X(vkCreateComputePipelines) X(vkCreateDescriptorPool) X(vkDestroyDescriptorPool) X(vkAllocateDescriptorSets) \
	X(vkUpdateDescriptorSets) X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer) \
	X(vkEndCommandBuffer) X(vkCmdBindPipeline) X(vkCmdBindDescriptorSets) X(vkCmdPushConstants) X(vkCmdDispatch) X(vkCmdPipelineBarrier) \
	X(vkQueueSubmit) X(vkResetCommandBuffer) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences) \
	X(vkCmdCopyBuffer) X(vkCreateSemaphore) X(vkDestroySemaphore) X(vkCreateQueryPool) X(vkDestroyQueryPool) X(vkCmdResetQueryPool) X(vkCmdWriteTimestamp) X(vkGetQueryPoolResults)

#define X(n) static PFN_##n n;
VK_FUNCS(X)
#undef X

namespace {

constexpr int kBindings = 16;

void chk(VkResult r, const char* what) {
	if (r == VK_ERROR_DEVICE_LOST) gpu_lost(std::string("Vulkan device lost while scoring VMAF (") + what + ")");
	if (r != VK_SUCCESS) throw std::runtime_error(std::string("Vulkan: ") + what + " failed (" + std::to_string((int)r) + ")");
}

// The instance and the list of devices, made once.
struct Instance {
	bool tried = false, ok = false;
	std::string why;
	VkInstance inst{};
	std::vector<VkPhysicalDevice> phys;
	std::vector<VkDeviceInfo> info;
};

// One device, shared by every user of it; made the first time it is asked for.
struct Shared {
	bool tried = false, ok = false;
	std::string why, name;
	VkPhysicalDevice phys{};
	VkDevice dev{};
	VkQueue queue{};
	uint32_t family = 0;
	// A second queue for copying frames in while the compute queue works on others (a transfer-only family, the DMA engine, where there is one; else a
	// second queue of the compute family); none if the device has neither, or VES_VMAF_NO_ASYNC_UPLOAD is set.
	VkQueue xqueue{};
	uint32_t xfamily = 0;
	bool has_x = false;
	// A second compute queue (of the compute family), for work that runs beside the main queue's (begin2()). Off unless VES_VMAF_SECOND_COMPUTE=1:
	// measured on an RTX 4080 and an AMD iGPU it gave no gain (up to 5% slower at 1080p); none either if the family has too few queues.
	VkQueue cqueue2{};
	bool has_c2 = false;
	VkPhysicalDeviceMemoryProperties mem{};
	double timestamp_ns = 0; // nanoseconds per timestamp tick; 0 if the compute queue has no timestamps
	VkDescriptorSetLayout dsl{};
	VkPipelineLayout layout{};
	std::mutex queue_lock, xqueue_lock, cqueue2_lock, pipe_lock;
	std::map<std::string, VkPipeline> pipelines;
};

std::mutex init_lock;
Instance instance;
std::map<int, std::unique_ptr<Shared>> devices_up;

bool load_loader(PFN_vkGetInstanceProcAddr& gipa, std::string& why) {
#ifdef _WIN32
	HMODULE lib = LoadLibraryA("vulkan-1.dll");
	if (!lib) { why = "no vulkan-1.dll"; return false; }
	gipa = (PFN_vkGetInstanceProcAddr)GetProcAddress(lib, "vkGetInstanceProcAddr");
#else
	void* lib = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_LOCAL);
	if (!lib) { why = "no libvulkan.so.1"; return false; }
	gipa = (PFN_vkGetInstanceProcAddr)dlsym(lib, "vkGetInstanceProcAddr");
#endif
	if (!gipa) { why = "no vkGetInstanceProcAddr"; return false; }
	return true;
}

// Loads Vulkan and lists its devices once; init_lock must be held.
bool instance_up(std::string& why) {
	Instance& in = instance;
	if (in.tried) { why = in.why; return in.ok; }
	in.tried = true;
	try {
		PFN_vkGetInstanceProcAddr gipa = nullptr;
		if (!load_loader(gipa, in.why)) { why = in.why; return false; }
		auto create_instance = (PFN_vkCreateInstance)gipa(nullptr, "vkCreateInstance");
		VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
		ai.apiVersion = VK_API_VERSION_1_2;
		VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
		ici.pApplicationInfo = &ai;
		chk(create_instance(&ici, nullptr, &in.inst), "vkCreateInstance");
#define X(n) n = (PFN_##n)gipa(in.inst, #n);
		VK_FUNCS(X)
#undef X
		uint32_t n = 0;
		vkEnumeratePhysicalDevices(in.inst, &n, nullptr);
		in.phys.resize(n);
		vkEnumeratePhysicalDevices(in.inst, &n, in.phys.data());
		in.phys.resize(n);
		for (uint32_t i = 0; i < n; i++) {
			VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
			VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
			p2.pNext = &idp;
			if (vkGetPhysicalDeviceProperties2) vkGetPhysicalDeviceProperties2(in.phys[i], &p2);
			else vkGetPhysicalDeviceProperties(in.phys[i], &p2.properties);
			VkPhysicalDeviceFeatures f;
			vkGetPhysicalDeviceFeatures(in.phys[i], &f);
			VkDeviceInfo d;
			d.index = (int)i;
			d.name = p2.properties.deviceName;
			d.discrete = p2.properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
			d.usable = f.shaderInt64 && f.shaderFloat64;
			d.vendor_id = p2.properties.vendorID;
			d.device_id = p2.properties.deviceID;
			if (vkGetPhysicalDeviceProperties2 && idp.deviceLUIDValid) { memcpy(&d.luid, idp.deviceLUID, sizeof d.luid); d.luid_valid = true; }
			VkPhysicalDeviceMemoryProperties mp;
			vkGetPhysicalDeviceMemoryProperties(in.phys[i], &mp);
			for (uint32_t h = 0; h < mp.memoryHeapCount; h++)
				if (mp.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) d.local_memory = std::max<uint64_t>(d.local_memory, mp.memoryHeaps[h].size);
			in.info.push_back(d);
		}
		in.ok = true;
	} catch (const std::exception& e) {
		in.why = e.what();
	}
	why = in.why;
	return in.ok;
}

// Brings device `index` up once; later calls return the first outcome. nullptr (and `why`) if it cannot be had.
Shared* bring_up(int index, std::string& why) {
	std::lock_guard lock(init_lock);
	if (!instance_up(why)) return nullptr;
	if (index < 0 || index >= (int)instance.info.size()) { why = "there is no Vulkan device " + std::to_string(index); return nullptr; }
	auto& slot = devices_up[index];
	if (!slot) slot = std::make_unique<Shared>();
	Shared& s = *slot;
	if (s.tried) { why = s.why; return s.ok ? &s : nullptr; }
	s.tried = true;
	try {
		if (!instance.info[index].usable) throw std::runtime_error("Vulkan device " + std::to_string(index) + " (" + instance.info[index].name + ") lacks 64 bit integer and float support");
		s.phys = instance.phys[index];
		s.name = instance.info[index].name;
		vkGetPhysicalDeviceMemoryProperties(s.phys, &s.mem);
		uint32_t qn = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(s.phys, &qn, nullptr);
		std::vector<VkQueueFamilyProperties> qf(qn);
		vkGetPhysicalDeviceQueueFamilyProperties(s.phys, &qn, qf.data());
		bool found = false;
		for (uint32_t i = 0; i < qn; i++) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { s.family = i; found = true; break; }
		if (!found) throw std::runtime_error("no compute queue");
		{
			VkPhysicalDeviceProperties props;
			vkGetPhysicalDeviceProperties(s.phys, &props);
			if (qf[s.family].timestampValidBits && props.limits.timestampPeriod > 0) s.timestamp_ns = props.limits.timestampPeriod;
		}
		const char* no_async = getenv("VES_VMAF_NO_ASYNC_UPLOAD");
		if (!(no_async && atoi(no_async) > 0)) {
			for (uint32_t i = 0; i < qn && !s.has_x; i++)
				if ((qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(qf[i].queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_GRAPHICS_BIT)) && qf[i].queueCount > 0) {
					s.xfamily = i;
					s.has_x = true;
				}
			if (!s.has_x && qf[s.family].queueCount >= 2) { s.xfamily = s.family; s.has_x = true; }
		}
		const char* c2 = getenv("VES_VMAF_SECOND_COMPUTE");
		const uint32_t used = 1 + (s.has_x && s.xfamily == s.family ? 1 : 0);
		if (c2 && atoi(c2) > 0 && qf[s.family].queueCount > used) s.has_c2 = true;
		float prio[3] = {1.0f, 1.0f, 1.0f};
		VkDeviceQueueCreateInfo qcis[2]{{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
		uint32_t qcount = 1;
		qcis[0].queueFamilyIndex = s.family;
		qcis[0].queueCount = used + (s.has_c2 ? 1 : 0);
		qcis[0].pQueuePriorities = prio;
		if (s.has_x && s.xfamily != s.family) {
			qcis[1].queueFamilyIndex = s.xfamily;
			qcis[1].queueCount = 1;
			qcis[1].pQueuePriorities = prio;
			qcount = 2;
		}
		VkPhysicalDeviceFeatures feats{};
		feats.shaderInt64 = VK_TRUE;
		feats.shaderFloat64 = VK_TRUE;
		VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
		dci.queueCreateInfoCount = qcount;
		dci.pQueueCreateInfos = qcis;
		dci.pEnabledFeatures = &feats;
		chk(vkCreateDevice(s.phys, &dci, nullptr, &s.dev), "vkCreateDevice");
		vkGetDeviceQueue(s.dev, s.family, 0, &s.queue);
		if (s.has_x) vkGetDeviceQueue(s.dev, s.xfamily, s.xfamily == s.family ? 1 : 0, &s.xqueue);
		if (s.has_c2) vkGetDeviceQueue(s.dev, s.family, used, &s.cqueue2);
		VkDescriptorSetLayoutBinding b[kBindings];
		for (int i = 0; i < kBindings; i++) b[i] = {(uint32_t)i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
		VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		dli.bindingCount = kBindings;
		dli.pBindings = b;
		chk(vkCreateDescriptorSetLayout(s.dev, &dli, nullptr, &s.dsl), "descriptor set layout");
		VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 128};
		VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		pli.setLayoutCount = 1;
		pli.pSetLayouts = &s.dsl;
		pli.pushConstantRangeCount = 1;
		pli.pPushConstantRanges = &pcr;
		chk(vkCreatePipelineLayout(s.dev, &pli, nullptr, &s.layout), "pipeline layout");
		s.ok = true;
	} catch (const std::exception& e) {
		s.why = e.what();
		s.ok = false;
	}
	why = s.why;
	return s.ok ? &s : nullptr;
}

} // namespace

struct Vk::Impl {
	Shared* s = nullptr;
	VkCommandPool pool{};
	struct Slot {
		VkCommandBuffer cmd{};
		VkFence fence{};
		VkQueryPool stamps{}; // kStamps timestamps: 0 the work began, 1 it ended, the rest for parts of it
		VkQueryPool pstamps{}; // VES_SCORE_TIMING=2: 0 the work began, then one after each dispatch, of the shader pnames[i - 1]
		std::vector<const char*> pnames;
		VkCommandBuffer xcmd{}; // copies on the second queue (upload_begin()), which the slot's next submit() waits for through xsem
		VkSemaphore xsem{};
		bool xpending = false;
		VkCommandBuffer cmd2{}; // work on the second compute queue (begin2()), with its own fence, which wait() waits for as well
		VkFence fence2{};
		VkSemaphore xsem2{};    // the copies, for the second compute queue
		bool xpending2 = false, pending2 = false;
	};
	VkCommandPool xpool{};
	std::vector<Slot> slots;
	VkCommandBuffer cmd{}; // the one being recorded (either compute queue)
	std::vector<std::pair<VkBuffer, std::vector<VkBuffer>>> aliases; // alias(): a placeholder, and the buffer of each slot
	int cur_slot = -1;     // ... its slot
	const char* cur_name = ""; // the shader bound last
	bool pass = false;     // VES_SCORE_TIMING=2
	bool second = false;   // ... and it is the second one's
	VkCommandBuffer xrec{}; // the one being recorded (second queue)
	VkDescriptorPool dpool{};
	std::vector<std::pair<VkBuffer, VkDeviceMemory>> owned;
	uint64_t device_bytes = 0, host_bytes = 0;
};

std::vector<VkDeviceInfo> Vk::devices(std::string& why) {
	std::lock_guard lock(init_lock);
	if (!instance_up(why)) return {};
	return instance.info;
}

bool Vk::available(int device, std::string& why) { return bring_up(device, why) != nullptr; }

Vk::Vk(int device, int slots) : impl(new Impl) {
	std::string why;
	impl->s = bring_up(device, why);
	if (!impl->s) { delete impl; impl = nullptr; throw std::runtime_error("Vulkan: " + why); }
	Shared& s = *impl->s;
	if (const char* e = getenv("VES_SCORE_TIMING"); e && atoi(e) >= 2) impl->pass = true;
	try {
		VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 * 1024};
		VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		dpi.maxSets = 1024; // aliased pipelines take a set per slot
		dpi.poolSizeCount = 1;
		dpi.pPoolSizes = &ps;
		chk(vkCreateDescriptorPool(s.dev, &dpi, nullptr, &impl->dpool), "descriptor pool");
		VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		cpi.queueFamilyIndex = s.family;
		chk(vkCreateCommandPool(s.dev, &cpi, nullptr, &impl->pool), "command pool");
		if (s.has_x) {
			VkCommandPoolCreateInfo xpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
			xpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
			xpi.queueFamilyIndex = s.xfamily;
			chk(vkCreateCommandPool(s.dev, &xpi, nullptr, &impl->xpool), "command pool (copies)");
		}
		impl->slots.resize(std::max(1, slots));
		for (auto& sl : impl->slots) {
			VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
			cai.commandPool = impl->pool;
			cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
			cai.commandBufferCount = 1;
			chk(vkAllocateCommandBuffers(s.dev, &cai, &sl.cmd), "command buffer");
			VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
			chk(vkCreateFence(s.dev, &fi, nullptr, &sl.fence), "fence");
			if (s.has_c2) {
				VkCommandBufferAllocateInfo cai2{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
				cai2.commandPool = impl->pool;
				cai2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				cai2.commandBufferCount = 1;
				chk(vkAllocateCommandBuffers(s.dev, &cai2, &sl.cmd2), "command buffer (second queue)");
				chk(vkCreateFence(s.dev, &fi, nullptr, &sl.fence2), "fence");
				if (s.has_x) {
					VkSemaphoreCreateInfo smi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
					chk(vkCreateSemaphore(s.dev, &smi, nullptr, &sl.xsem2), "semaphore");
				}
			}
			if (s.has_x) {
				VkCommandBufferAllocateInfo xai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
				xai.commandPool = impl->xpool;
				xai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
				xai.commandBufferCount = 1;
				chk(vkAllocateCommandBuffers(s.dev, &xai, &sl.xcmd), "command buffer (copies)");
				VkSemaphoreCreateInfo smi{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
				chk(vkCreateSemaphore(s.dev, &smi, nullptr, &sl.xsem), "semaphore");
			}
			if (s.timestamp_ns > 0) {
				VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
				qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
				qi.queryCount = kStamps;
				chk(vkCreateQueryPool(s.dev, &qi, nullptr, &sl.stamps), "query pool");
				if (impl->pass) {
					qi.queryCount = kPassStamps;
					chk(vkCreateQueryPool(s.dev, &qi, nullptr, &sl.pstamps), "query pool (passes)");
				}
			}
		}
	} catch (...) {
		release();
		throw;
	}
}

Vk::~Vk() { release(); }

void Vk::release() {
	if (!impl) return;
	Shared& s = *impl->s;
	for (auto& sl : impl->slots) {
		if (sl.fence) vkDestroyFence(s.dev, sl.fence, nullptr);
		if (sl.stamps) vkDestroyQueryPool(s.dev, sl.stamps, nullptr);
		if (sl.pstamps) vkDestroyQueryPool(s.dev, sl.pstamps, nullptr);
		if (sl.xsem) vkDestroySemaphore(s.dev, sl.xsem, nullptr);
		if (sl.xsem2) vkDestroySemaphore(s.dev, sl.xsem2, nullptr);
		if (sl.fence2) vkDestroyFence(s.dev, sl.fence2, nullptr);
	}
	if (impl->pool) vkDestroyCommandPool(s.dev, impl->pool, nullptr);
	if (impl->xpool) vkDestroyCommandPool(s.dev, impl->xpool, nullptr);
	if (impl->dpool) vkDestroyDescriptorPool(s.dev, impl->dpool, nullptr);
	for (auto& [b, m] : impl->owned) { vkDestroyBuffer(s.dev, b, nullptr); vkFreeMemory(s.dev, m, nullptr); }
	delete impl;
	impl = nullptr;
}

Buf Vk::buffer(VkDeviceSize size, int kind) {
	Shared& s = *impl->s;
	Buf b;
	b.size = size;
	VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bi.size = size;
	bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	// With a second queue of another family, every buffer may be used by both (the copies in and the compute work), without ownership transfers.
	const uint32_t families[2] = {s.family, s.xfamily};
	if (s.has_x && s.xfamily != s.family) {
		bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
		bi.queueFamilyIndexCount = 2;
		bi.pQueueFamilyIndices = families;
	}
	chk(vkCreateBuffer(s.dev, &bi, nullptr, &b.buf), "vkCreateBuffer");
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(s.dev, b.buf, &mr);
	const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, dl = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, hc = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
	// kind 3: host visible and not device local where there is such memory (system memory the GPU copies from), else like kind 0
	const VkMemoryPropertyFlags want[4][3] = {{hv | dl, hv, 0}, {dl, 0, 0}, {hv | hc, hv, 0}, {hv, hv, 0}};
	const VkMemoryPropertyFlags avoid[4][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}, {dl, 0, 0}};
	// The memory types in order of preference; when one is out of memory (a small window of device-local host-visible memory without resizable BAR),
	// the next one is tried.
	std::vector<uint32_t> types;
	for (int w = 0; w < 3; w++)
		for (uint32_t i = 0; i < s.mem.memoryTypeCount; i++)
			if ((mr.memoryTypeBits & (1u << i)) && (s.mem.memoryTypes[i].propertyFlags & want[kind][w]) == want[kind][w] && !(s.mem.memoryTypes[i].propertyFlags & avoid[kind][w]) &&
			    (want[kind][w] || w == 2) && std::find(types.begin(), types.end(), i) == types.end())
				types.push_back(i);
	if (types.empty()) { vkDestroyBuffer(s.dev, b.buf, nullptr); throw std::runtime_error("Vulkan: no suitable memory type"); }
	int found = -1;
	VkResult r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
	for (uint32_t t : types) {
		VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		mi.allocationSize = mr.size;
		mi.memoryTypeIndex = t;
		r = vkAllocateMemory(s.dev, &mi, nullptr, &b.mem);
		if (r == VK_SUCCESS) { found = (int)t; break; }
		if (r != VK_ERROR_OUT_OF_DEVICE_MEMORY && r != VK_ERROR_OUT_OF_HOST_MEMORY) break;
	}
	if (found < 0) { vkDestroyBuffer(s.dev, b.buf, nullptr); chk(r, "vkAllocateMemory"); }
	if (kind != 1 && !(s.mem.memoryTypes[found].propertyFlags & hv)) {
		vkFreeMemory(s.dev, b.mem, nullptr);
		vkDestroyBuffer(s.dev, b.buf, nullptr);
		throw std::runtime_error("Vulkan: out of host visible memory");
	}
	impl->owned.emplace_back(b.buf, b.mem);
	(s.mem.memoryTypes[found].propertyFlags & dl ? impl->device_bytes : impl->host_bytes) += mr.size;
	chk(vkBindBufferMemory(s.dev, b.buf, b.mem, 0), "vkBindBufferMemory");
	// Host visible memory is taken coherent here; the memory types picked above are on every desktop GPU.
	if (s.mem.memoryTypes[found].propertyFlags & hv) {
		if (!(s.mem.memoryTypes[found].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) throw std::runtime_error("Vulkan: host memory is not coherent");
		chk(vkMapMemory(s.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map), "vkMapMemory");
	}
	return b;
}

Pipe Vk::pipeline(const char* name, const std::vector<Buf*>& bindings) {
	Shared& s = *impl->s;
	Pipe p;
	{
		std::lock_guard lock(s.pipe_lock);
		auto it = s.pipelines.find(name);
		if (it == s.pipelines.end()) {
			size_t size = 0;
			const unsigned char* code = spv_find(name, size);
			if (!code) throw std::runtime_error(std::string("Vulkan: no shader ") + name);
			VkShaderModuleCreateInfo si{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
			si.codeSize = size;
			si.pCode = (const uint32_t*)code;
			VkShaderModule sm;
			chk(vkCreateShaderModule(s.dev, &si, nullptr, &sm), "shader module");
			VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
			pi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr};
			pi.layout = s.layout;
			VkPipeline pipe;
			chk(vkCreateComputePipelines(s.dev, VK_NULL_HANDLE, 1, &pi, nullptr, &pipe), name);
			it = s.pipelines.emplace(name, pipe).first;
		}
		p.pipe = it->second;
		p.name = it->first.c_str(); // the map's nodes stay put
	}
	VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
	ai.descriptorPool = impl->dpool;
	ai.descriptorSetCount = 1;
	ai.pSetLayouts = &s.dsl;
	chk(vkAllocateDescriptorSets(s.dev, &ai, &p.set), "descriptor set");
	std::vector<VkDescriptorBufferInfo> infos(kBindings);
	std::vector<VkWriteDescriptorSet> writes;
	for (int i = 0; i < kBindings; i++) {
		Buf* b = i < (int)bindings.size() && bindings[i] ? bindings[i] : bindings[0]; // unused bindings point at the first buffer
		infos[i] = {b->buf, 0, VK_WHOLE_SIZE};
		VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		w.dstSet = p.set;
		w.dstBinding = i;
		w.descriptorCount = 1;
		w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		w.pBufferInfo = &infos[i];
		writes.push_back(w);
	}
	vkUpdateDescriptorSets(s.dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);
	bool aliased = false;
	for (auto& info : infos) for (auto& [from, to] : impl->aliases) if (info.buffer == from) aliased = true;
	if (aliased) {
		p.nslot = (int)impl->slots.size();
		if (p.nslot > 8) throw std::runtime_error("Vulkan: too many slots for aliased buffers");
		for (int k = 0; k < p.nslot; k++) {
			chk(vkAllocateDescriptorSets(s.dev, &ai, &p.slot_sets[k]), "descriptor set");
			std::vector<VkDescriptorBufferInfo> si = infos;
			for (auto& info : si) for (auto& [from, to] : impl->aliases) if (info.buffer == from) info.buffer = to[k];
			for (int i = 0; i < kBindings; i++) { writes[i].dstSet = p.slot_sets[k]; writes[i].pBufferInfo = &si[i]; }
			vkUpdateDescriptorSets(s.dev, (uint32_t)writes.size(), writes.data(), 0, nullptr);
		}
	}
	return p;
}

void Vk::alias(const Buf& placeholder, const std::vector<const Buf*>& per_slot) {
	if (per_slot.size() != impl->slots.size()) throw std::runtime_error("Vulkan: alias needs a buffer per slot");
	std::vector<VkBuffer> to;
	for (const Buf* b : per_slot) to.push_back(b->buf);
	impl->aliases.emplace_back(placeholder.buf, to);
}

void Vk::begin(int slot) {
	Impl::Slot& sl = impl->slots.at(slot);
	impl->cmd = sl.cmd;
	impl->second = false;
	vkResetCommandBuffer(impl->cmd, 0);
	VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	chk(vkBeginCommandBuffer(impl->cmd, &bi), "begin");
	if (sl.stamps) vkCmdResetQueryPool(impl->cmd, sl.stamps, 0, kStamps);
	impl->cur_slot = slot;
	if (sl.pstamps) {
		vkCmdResetQueryPool(impl->cmd, sl.pstamps, 0, kPassStamps);
		vkCmdWriteTimestamp(impl->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sl.pstamps, 0);
		sl.pnames.clear();
	}
}

void Vk::stamp(int slot, int which) {
	Impl::Slot& sl = impl->slots.at(slot);
	if (impl->second) return; // the stamps time the main queue's work only
	// bottom of pipe: written once every command before it, in this submission and the ones before on the queue, is done
	if (sl.stamps) vkCmdWriteTimestamp(impl->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sl.stamps, (uint32_t)which);
}

void Vk::copy(const Buf& src, const Buf& dst, VkDeviceSize size) {
	VkBufferCopy c{0, 0, size ? size : std::min(src.size, dst.size)};
	vkCmdCopyBuffer(impl->cmd, src.buf, dst.buf, 1, &c);
}

void Vk::copy_range(const Buf& src, VkDeviceSize src_off, const Buf& dst, VkDeviceSize dst_off, VkDeviceSize size) {
	VkBufferCopy c{src_off, dst_off, size};
	vkCmdCopyBuffer(impl->cmd, src.buf, dst.buf, 1, &c);
}

void Vk::bind(const Pipe& p) {
	impl->cur_name = p.name;
	vkCmdBindPipeline(impl->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
	const VkDescriptorSet set = p.nslot > 0 && impl->cur_slot >= 0 && impl->cur_slot < p.nslot ? p.slot_sets[impl->cur_slot] : p.set;
	vkCmdBindDescriptorSets(impl->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl->s->layout, 0, 1, &set, 0, nullptr);
}

void Vk::push(const void* data, uint32_t size) { vkCmdPushConstants(impl->cmd, impl->s->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data); }

void Vk::dispatch(uint32_t x, uint32_t y) {
	vkCmdDispatch(impl->cmd, x, y, 1);
	if (!impl->pass || impl->second || impl->cur_slot < 0) return;
	Impl::Slot& sl = impl->slots[impl->cur_slot];
	if (!sl.pstamps || sl.pnames.size() + 1 >= (size_t)kPassStamps) return;
	sl.pnames.push_back(impl->cur_name);
	vkCmdWriteTimestamp(impl->cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, sl.pstamps, (uint32_t)sl.pnames.size());
}

bool Vk::pass_timing() const { return impl->pass && impl->s->timestamp_ns > 0; }

void Vk::pass_seconds(int slot, std::vector<std::pair<const char*, double>>& out) {
	out.clear();
	Impl::Slot& sl = impl->slots.at(slot);
	if (!sl.pstamps || sl.pnames.empty()) return;
	std::vector<uint64_t> t(sl.pnames.size() + 1);
	if (vkGetQueryPoolResults(impl->s->dev, sl.pstamps, 0, (uint32_t)t.size(), t.size() * 8, t.data(), 8, VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return;
	for (size_t i = 0; i < sl.pnames.size(); i++)
		out.emplace_back(sl.pnames[i], t[i + 1] >= t[i] ? (double)(t[i + 1] - t[i]) * impl->s->timestamp_ns * 1e-9 : 0.0);
}

void Vk::barrier() {
	VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
	vkCmdPipelineBarrier(impl->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

bool Vk::async_upload() const { return impl->s->has_x; }

void Vk::upload_begin(int slot) {
	Impl::Slot& sl = impl->slots.at(slot);
	impl->xrec = sl.xcmd;
	vkResetCommandBuffer(impl->xrec, 0);
	VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	chk(vkBeginCommandBuffer(impl->xrec, &bi), "begin (copies)");
}

void Vk::upload_copy(const Buf& src, const Buf& dst, VkDeviceSize size) {
	VkBufferCopy c{0, 0, size ? size : std::min(src.size, dst.size)};
	vkCmdCopyBuffer(impl->xrec, src.buf, dst.buf, 1, &c);
}

void Vk::upload_submit(int slot, bool for_second) {
	Shared& s = *impl->s;
	Impl::Slot& sl = impl->slots.at(slot);
	chk(vkEndCommandBuffer(sl.xcmd), "end (copies)");
	VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &sl.xcmd;
	const VkSemaphore sems[2] = {sl.xsem, sl.xsem2};
	for_second = for_second && sl.xsem2;
	si.signalSemaphoreCount = for_second ? 2 : 1;
	si.pSignalSemaphores = sems;
	{
		std::lock_guard lock(s.xqueue_lock);
		chk(vkQueueSubmit(s.xqueue, 1, &si, VK_NULL_HANDLE), "submit (copies)");
	}
	sl.xpending = true;
	sl.xpending2 = for_second;
}

bool Vk::second_compute() const { return impl->s->has_c2; }

void Vk::begin2(int slot) {
	Impl::Slot& sl = impl->slots.at(slot);
	impl->cmd = sl.cmd2;
	impl->second = true;
	vkResetCommandBuffer(impl->cmd, 0);
	VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	chk(vkBeginCommandBuffer(impl->cmd, &bi), "begin (second queue)");
}

void Vk::submit2(int slot) {
	Shared& s = *impl->s;
	Impl::Slot& sl = impl->slots.at(slot);
	chk(vkEndCommandBuffer(sl.cmd2), "end (second queue)");
	VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &sl.cmd2;
	const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	if (sl.xpending2) {
		si.waitSemaphoreCount = 1;
		si.pWaitSemaphores = &sl.xsem2;
		si.pWaitDstStageMask = &wait_stage;
		sl.xpending2 = false;
	}
	vkResetFences(s.dev, 1, &sl.fence2);
	{
		std::lock_guard lock(s.cqueue2_lock);
		chk(vkQueueSubmit(s.cqueue2, 1, &si, sl.fence2), "submit (second queue)");
	}
	sl.pending2 = true;
	impl->second = false;
}

void Vk::submit(int slot) {
	Shared& s = *impl->s;
	Impl::Slot& sl = impl->slots.at(slot);
	chk(vkEndCommandBuffer(sl.cmd), "end");
	VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &sl.cmd;
	// the copies of upload_submit() are done before this work starts; the slot's fence then also means they are done
	const VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
	if (sl.xpending) {
		si.waitSemaphoreCount = 1;
		si.pWaitSemaphores = &sl.xsem;
		si.pWaitDstStageMask = &wait_stage;
		sl.xpending = false;
	}
	vkResetFences(s.dev, 1, &sl.fence);
	std::lock_guard lock(s.queue_lock);
	chk(vkQueueSubmit(s.queue, 1, &si, sl.fence), "submit");
}

void Vk::memory(uint64_t& device_local, uint64_t& host) const { device_local = impl->device_bytes; host = impl->host_bytes; }

void Vk::wait(int slot) {
	Impl::Slot& sl = impl->slots.at(slot);
	chk(vkWaitForFences(impl->s->dev, 1, &sl.fence, VK_TRUE, ~0ull), "wait");
	if (sl.pending2) {
		chk(vkWaitForFences(impl->s->dev, 1, &sl.fence2, VK_TRUE, ~0ull), "wait (second queue)");
		sl.pending2 = false;
	}
}

double Vk::gpu_seconds(int slot, int from, int to) {
	Impl::Slot& sl = impl->slots.at(slot);
	if (!sl.stamps || from < 0 || to < 0 || from >= kStamps || to >= kStamps) return -1;
	uint64_t t[2] = {0, 0};
	// one at a time: a stamp this submission did not write is not ready (VK_NOT_READY), and then there is no time for it
	if (vkGetQueryPoolResults(impl->s->dev, sl.stamps, (uint32_t)from, 1, sizeof t[0], &t[0], sizeof t[0], VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return -1;
	if (vkGetQueryPoolResults(impl->s->dev, sl.stamps, (uint32_t)to, 1, sizeof t[1], &t[1], sizeof t[1], VK_QUERY_RESULT_64_BIT) != VK_SUCCESS) return -1;
	if (t[1] < t[0]) return -1;
	return (double)(t[1] - t[0]) * impl->s->timestamp_ns * 1e-9;
}
#endif
