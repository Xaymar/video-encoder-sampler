#include "vk_compute.h"

#ifndef VES_NO_VULKAN
#include <algorithm>
#include <cstdio>
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
	X(vkQueueSubmit) X(vkResetCommandBuffer) X(vkCreateFence) X(vkDestroyFence) X(vkWaitForFences) X(vkResetFences)

#define X(n) static PFN_##n n;
VK_FUNCS(X)
#undef X

namespace {

constexpr int kBindings = 16;

void chk(VkResult r, const char* what) {
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
	VkPhysicalDeviceMemoryProperties mem{};
	VkDescriptorSetLayout dsl{};
	VkPipelineLayout layout{};
	std::mutex queue_lock, pipe_lock;
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
		float prio = 1.0f;
		VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
		qci.queueFamilyIndex = s.family;
		qci.queueCount = 1;
		qci.pQueuePriorities = &prio;
		VkPhysicalDeviceFeatures feats{};
		feats.shaderInt64 = VK_TRUE;
		feats.shaderFloat64 = VK_TRUE;
		VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
		dci.queueCreateInfoCount = 1;
		dci.pQueueCreateInfos = &qci;
		dci.pEnabledFeatures = &feats;
		chk(vkCreateDevice(s.phys, &dci, nullptr, &s.dev), "vkCreateDevice");
		vkGetDeviceQueue(s.dev, s.family, 0, &s.queue);
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
	VkCommandBuffer cmd{};
	VkFence fence{};
	VkDescriptorPool dpool{};
	std::vector<std::pair<VkBuffer, VkDeviceMemory>> owned;
};

std::vector<VkDeviceInfo> Vk::devices(std::string& why) {
	std::lock_guard lock(init_lock);
	if (!instance_up(why)) return {};
	return instance.info;
}

bool Vk::available(int device, std::string& why) { return bring_up(device, why) != nullptr; }

Vk::Vk(int device) : impl(new Impl) {
	std::string why;
	impl->s = bring_up(device, why);
	if (!impl->s) { delete impl; impl = nullptr; throw std::runtime_error("Vulkan: " + why); }
	Shared& s = *impl->s;
	try {
		VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 * 256};
		VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		dpi.maxSets = 256;
		dpi.poolSizeCount = 1;
		dpi.pPoolSizes = &ps;
		chk(vkCreateDescriptorPool(s.dev, &dpi, nullptr, &impl->dpool), "descriptor pool");
		VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		cpi.queueFamilyIndex = s.family;
		chk(vkCreateCommandPool(s.dev, &cpi, nullptr, &impl->pool), "command pool");
		VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		cai.commandPool = impl->pool;
		cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		cai.commandBufferCount = 1;
		chk(vkAllocateCommandBuffers(s.dev, &cai, &impl->cmd), "command buffer");
		VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		chk(vkCreateFence(s.dev, &fi, nullptr, &impl->fence), "fence");
	} catch (...) {
		release();
		throw;
	}
}

Vk::~Vk() { release(); }

void Vk::release() {
	if (!impl) return;
	Shared& s = *impl->s;
	if (impl->fence) vkDestroyFence(s.dev, impl->fence, nullptr);
	if (impl->pool) vkDestroyCommandPool(s.dev, impl->pool, nullptr);
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
	chk(vkCreateBuffer(s.dev, &bi, nullptr, &b.buf), "vkCreateBuffer");
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(s.dev, b.buf, &mr);
	const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, dl = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, hc = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
	const VkMemoryPropertyFlags want[3][3] = {{hv | dl, hv, 0}, {dl, 0, 0}, {hv | hc, hv, 0}};
	int found = -1;
	for (int w = 0; w < 3 && found < 0; w++)
		for (uint32_t i = 0; i < s.mem.memoryTypeCount; i++)
			if ((mr.memoryTypeBits & (1u << i)) && (s.mem.memoryTypes[i].propertyFlags & want[kind][w]) == want[kind][w] && (want[kind][w] || w == 2)) { found = (int)i; break; }
	if (found < 0) { vkDestroyBuffer(s.dev, b.buf, nullptr); throw std::runtime_error("Vulkan: no suitable memory type"); }
	VkMemoryAllocateInfo mi{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	mi.allocationSize = mr.size;
	mi.memoryTypeIndex = found;
	VkResult r = vkAllocateMemory(s.dev, &mi, nullptr, &b.mem);
	if (r != VK_SUCCESS) { vkDestroyBuffer(s.dev, b.buf, nullptr); chk(r, "vkAllocateMemory"); }
	impl->owned.emplace_back(b.buf, b.mem);
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
	return p;
}

void Vk::begin() {
	vkResetCommandBuffer(impl->cmd, 0);
	VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	chk(vkBeginCommandBuffer(impl->cmd, &bi), "begin");
}

void Vk::bind(const Pipe& p) {
	vkCmdBindPipeline(impl->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipe);
	vkCmdBindDescriptorSets(impl->cmd, VK_PIPELINE_BIND_POINT_COMPUTE, impl->s->layout, 0, 1, &p.set, 0, nullptr);
}

void Vk::push(const void* data, uint32_t size) { vkCmdPushConstants(impl->cmd, impl->s->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, size, data); }

void Vk::dispatch(uint32_t x, uint32_t y) { vkCmdDispatch(impl->cmd, x, y, 1); }

void Vk::barrier() {
	VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
	mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
	mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_HOST_READ_BIT;
	vkCmdPipelineBarrier(impl->cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, nullptr, 0, nullptr);
}

void Vk::submit_wait() {
	Shared& s = *impl->s;
	chk(vkEndCommandBuffer(impl->cmd), "end");
	VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &impl->cmd;
	vkResetFences(s.dev, 1, &impl->fence);
	{
		std::lock_guard lock(s.queue_lock);
		chk(vkQueueSubmit(s.queue, 1, &si, impl->fence), "submit");
	}
	chk(vkWaitForFences(s.dev, 1, &impl->fence, VK_TRUE, ~0ull), "wait");
}
#endif
