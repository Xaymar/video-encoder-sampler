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
	const char* name = ""; // the shader's (pipeline()'s name, which must outlive it: a string literal)
};

class Vk {
public:
	// Every Vulkan device; empty (and `why`) when Vulkan cannot be loaded.
	static std::vector<VkDeviceInfo> devices(std::string& why);
	// Whether device `device` (an index into devices()) can run the scorer; `why` says what is missing when it cannot.
	static bool available(int device, std::string& why);

	// `slots`: how many command buffers can be recorded and in flight at once, each with its own fence (throws std::runtime_error if the device cannot be had).
	explicit Vk(int device, int slots = 1);
	~Vk();
	Vk(const Vk&) = delete;
	Vk& operator=(const Vk&) = delete;

	// kind 0: host visible, device local where there is a window for it (written by the CPU straight into VRAM); 1: device local; 2: host cached (read back
	// by the CPU); 3: host visible in system memory where there is such (written by the CPU, copied by the GPU). A kind that has run out of its first choice
	// of memory gets its next one.
	Buf buffer(VkDeviceSize size, int kind);
	// A compute pipeline from the embedded SPIR-V `name` (shared by all users), with a descriptor set of its own that points at `bindings` (binding i -> bindings[i]).
	Pipe pipeline(const char* name, const std::vector<Buf*>& bindings);
	// Recording into the command buffer of a slot (whose earlier work must be done: wait()), then submitting it. Submissions to a device run in order
	// of submission on its one queue; a barrier() also orders the commands after it against everything submitted before (other slots, other users).
	void begin(int slot = 0);
	void bind(const Pipe& p);
	void push(const void* data, uint32_t size);
	void dispatch(uint32_t x, uint32_t y = 1);
	void copy(const Buf& src, const Buf& dst, VkDeviceSize size = 0); // size 0: the smaller of the two buffers
	void copy_range(const Buf& src, VkDeviceSize src_off, const Buf& dst, VkDeviceSize dst_off, VkDeviceSize size);
	void barrier();
	static constexpr int kStamps = 12;
	void stamp(int slot, int which); // timestamp `which` (0: start, 1: end of the slot's work; up to kStamps - 1 for parts of it), see gpu_seconds()
	// Copies on a second queue (async_upload(): the device has one), so that copying a frame in overlaps the compute work on frames before it:
	// upload_begin() records into the slot's own command buffer for it, upload_submit() submits it, and the slot's next submit() waits for it.
	bool async_upload() const;
	void upload_begin(int slot);
	void upload_copy(const Buf& src, const Buf& dst, VkDeviceSize size = 0);
	void upload_submit(int slot, bool for_second = false); // for_second: the slot's next submit2() waits for these copies too
	// A second compute queue (second_compute(): the device has one), for work that needs nothing from the main queue's and can run beside it:
	// begin2() records the slot's work for it (with bind/push/dispatch/copy/barrier as usual; stamp() does nothing there), submit2() submits it, and
	// wait() waits for it as well. Work on it runs in submission order only against itself.
	bool second_compute() const;
	void begin2(int slot);
	void submit2(int slot);
	void submit(int slot = 0);
	void wait(int slot = 0); // until the slot's last submission is done
	// GPU time between two of the slot's stamps (by default start and end) in its last finished submission; -1 if the device has no timestamps or
	// the submission did not write both. Safe to call from another thread than the one that records, for a slot that is not being recorded.
	double gpu_seconds(int slot, int from = 0, int to = 1);
	// VES_SCORE_TIMING=2: a timestamp after every dispatch on the main queue (up to kPassStamps per submission). The GPU time of each dispatch of
	// the slot's last finished submission, in order, with its shader's name (from the end of the one before; the first from the start).
	static constexpr int kPassStamps = 1024;
	bool pass_timing() const;
	void pass_seconds(int slot, std::vector<std::pair<const char*, double>>& out);
	// Bytes of the buffers made so far, in device-local memory (VRAM, or what an integrated GPU sets aside) and in other memory.
	void memory(uint64_t& device_local, uint64_t& host) const;

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
