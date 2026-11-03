#pragma once
// A minimal Vulkan compute layer for the GPU VMAF scorer (vk_vmaf.cpp). One Vulkan device is shared by every scorer in the process; each `Vk`
// object is one user of it, with its own buffers, descriptor sets and command buffer, all released when it goes away.
#include <cstdint>
#include <string>
#include <vector>

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
	// Whether the device can run the scorer (a GPU with 64 bit integers and floats); `why` says what is missing when it cannot. `gpu`: index among the Vulkan devices, -1 for the first discrete one.
	static bool available(std::string& why, int gpu = -1);
	static std::string device_name();

	Vk();   // throws std::runtime_error if the device cannot be had
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
	static bool available(std::string& why, int = -1) { why = "built without Vulkan"; return false; }
	static std::string device_name() { return ""; }
};
#endif
