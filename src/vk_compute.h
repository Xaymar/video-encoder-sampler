#pragma once
// A minimal Vulkan compute layer for the GPU VMAF scorer (vk_vmaf.cpp). Each Vulkan device that is used is brought up once and shared by every
// scorer in the process that uses it; each `Vk` object is one user of one device, with its own buffers, descriptor sets and command buffer, all released when it goes away.
#include <cstdint>
#include <string>
#include <vector>

// A Vulkan device as the loader lists it.
struct VkDeviceInfo {
	int index = 0;                 // in the loader's order: what a number in vmaf.gpu means
	std::string name;
	bool discrete = false;
	bool usable = false;           // has the 64 bit integers and floats the scorer needs
	uint32_t vendor_id = 0, device_id = 0;
	uint64_t luid = 0;             // the adapter LUID (Windows), the same number as GpuInfo::luid
	bool luid_valid = false;
	uint64_t local_memory = 0;     // bytes in its largest device-local heap (an integrated GPU's is carved out of system memory)
};

// The Vulkan device that is the n-th GPU of a vendor as gpu.h lists them (matched by LUID where the driver says), -1 if none; and a device's name.
int vulkan_device_of_gpu(unsigned vendor, int n);
std::string vulkan_device_name(int index);
// The Vulkan devices from fastest to slowest, as far as one can tell without measuring: discrete GPUs first, then by device-local memory, then in the
// loader's order. Decoding, scaling and scoring run on the first one that works unless a setting names another (they do not need the encoder's GPU).
std::vector<VkDeviceInfo> vulkan_devices_fastest_first();
// The GPU behind Vulkan device `index` as gpu.h numbers it (vendor and the n-th adapter of that vendor), for decoding on it. False if it is none of them.
bool gpu_of_vulkan_device(int index, unsigned& vendor, int& n);

#ifndef VES_NO_VULKAN
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

struct Buf {
	VkBuffer buf = VK_NULL_HANDLE;
	VkDeviceMemory mem = VK_NULL_HANDLE;
	void* map = nullptr; // set for host visible memory
	VkDeviceSize size = 0;
};

struct Pipe {
	VkPipeline pipe = VK_NULL_HANDLE;
	VkDescriptorSet set = VK_NULL_HANDLE;
};

class Vk {
public:
	// Every Vulkan device; empty (and `why`) when Vulkan cannot be loaded.
	static std::vector<VkDeviceInfo> devices(std::string& why);
	// Whether device `device` (an index into devices()) can run the scorer; `why` says what is missing when it cannot.
	static bool available(int device, std::string& why);

	explicit Vk(int device); // throws std::runtime_error if the device cannot be had
	~Vk();
	Vk(const Vk&) = delete;
	Vk& operator=(const Vk&) = delete;

	// kind 0: host visible, device local where there is a window for it (written by the CPU straight into VRAM); 1: device local; 2: host cached (read back by the CPU)
	Buf buffer(VkDeviceSize size, int kind);
	// A compute pipeline from the embedded SPIR-V `name` (shared by all users), with a descriptor set of its own that points at `bindings` (binding i -> bindings[i]).
	Pipe pipeline(const char* name, const std::vector<Buf*>& bindings);
	void begin();
	void bind(const Pipe& p);
	void push(const void* data, uint32_t size);
	void dispatch(uint32_t x, uint32_t y = 1);
	void barrier();
	void submit_wait();

private:
	void release();
	struct Impl;
	Impl* impl;
};
#else
class Vk {
public:
	static std::vector<VkDeviceInfo> devices(std::string& why) { why = "built without Vulkan"; return {}; }
	static bool available(int, std::string& why) { why = "built without Vulkan"; return false; }
};
#endif
