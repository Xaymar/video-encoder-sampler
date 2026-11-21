#pragma once
// A minimal Vulkan compute layer for the GPU VMAF scorer (vk_vmaf.cpp). Each Vulkan device that is used is brought up once and shared by every
// scorer in the process that uses it; each `Vk` object is one user of one device, with its own buffers, descriptor sets and command buffer, all released when it goes away.
#include <cstdint>
#include <functional>
#include <memory>
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
	uint8_t uuid[16]{};            // VkPhysicalDeviceIDProperties::deviceUUID: the same number CUDA has for the GPU (cuDeviceGetUuid)
	bool uuid_valid = false;
};

// The Vulkan device that is the n-th GPU of a vendor as gpu.h lists them (matched by LUID where the driver says), -1 if none; and a device's name.
int vulkan_device_of_gpu(unsigned vendor, int n);
std::string vulkan_device_name(int index);
// The Vulkan devices from fastest to slowest, as far as one can tell without measuring: discrete GPUs first, then by device-local memory, then in the
// loader's order. Decoding, scaling and scoring run on the first one that works unless a setting names another (they do not need the encoder's GPU).
std::vector<VkDeviceInfo> vulkan_devices_fastest_first();
// The GPU behind Vulkan device `index` as gpu.h numbers it (vendor and the n-th adapter of that vendor), for decoding on it. False if it is none of them.
bool gpu_of_vulkan_device(int index, unsigned& vendor, int& n);
// The process's home GPU: the Vulkan device it decodes on (the reference and the scored streams). Converting and scoring go there too unless a
// setting names another GPU, so that the frames stay on it. -1 while there is none (then each picks the fastest).
void set_home_vulkan_device(int index);
int home_vulkan_device();

#ifndef VES_NO_VULKAN
#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

// A Vulkan device made elsewhere for Vk to use instead of making its own: FFmpeg's, so that frames its decoder made on the GPU can be read where they
// are (see ff_vulkan.cpp). It must have shaderInt64, shaderFloat64 and timeline semaphores on. `lock`/`unlock` guard a queue that its maker uses too.
struct VkExternal {
	VkPhysicalDevice phys = VK_NULL_HANDLE;
	VkDevice dev = VK_NULL_HANDLE;
	uint32_t compute_family = 0, compute_queues = 0;
	bool has_transfer = false; // a transfer-only family (for copies beside the compute work)
	uint32_t transfer_family = 0;
	uint32_t queue_flags = 0; // VkDeviceQueueCreateFlags its queues were made with (then they are fetched with vkGetDeviceQueue2)
	VkInstance inst = VK_NULL_HANDLE;
	std::vector<std::string> extensions; // the device extensions it was made with
	bool export_memory = false; // made with VK_KHR_external_memory_win32 (Windows) / VK_KHR_external_memory_fd: Vk::exportable_buffer() works
	void (*lock)(void* ctx, uint32_t family, uint32_t index) = nullptr;
	void (*unlock)(void* ctx, uint32_t family, uint32_t index) = nullptr;
	void* ctx = nullptr;
};
// Where devices come from when one is first brought up: `provider(index, vendor, device, out)` fills `out` for Vulkan device `index` (as Vk::devices()
// numbers it, with these PCI ids) and returns true, or returns false to have Vk make its own. Set before any device is brought up.
void vk_set_external_provider(bool (*provider)(int index, uint32_t vendor_id, uint32_t device_id, VkExternal& out));

// The native handles of Vulkan device `index` as Vk brought it up (bringing it up if needed), for a library that works on the same VkDevice (AMF,
// which takes frames there as Vulkan surfaces). False when it cannot be had or did not come from a VkExternal provider (FFmpeg's device): only that
// one is guaranteed to hold the frames. `compute_queues` is how many queues the compute family has (the scorer and the converter use the first);
// `first_unused_queue` is the first of them that Vk does not submit to itself (the second one holds the copies when the device has no transfer family).
struct VkNative {
	VkInstance inst = VK_NULL_HANDLE;
	VkPhysicalDevice phys = VK_NULL_HANDLE;
	VkDevice dev = VK_NULL_HANDLE;
	uint32_t compute_family = 0, compute_queues = 0, first_unused_queue = 1;
	std::vector<std::string> extensions;
};
bool vk_native(int index, VkNative& out);

// A plane of an image on the device (a frame decoded there) to be copied into a buffer: `aspect` is VK_IMAGE_ASPECT_PLANE_n_BIT of a multi-planar
// image or VK_IMAGE_ASPECT_COLOR_BIT; w x h texels.
struct VkImagePlane {
	VkImage img = VK_NULL_HANDLE;
	VkImageAspectFlags aspect = 0;
	uint32_t w = 0, h = 0;
};

// A frame decoded on a device that Vk uses (an FFmpeg Vulkan frame, see ff_vulkan.cpp), described for reading it where it is. The planes are those
// of the decoder's format: Y, then either one plane of interleaved U and V (NV12, P010) or U and V; samples `bytes` wide with the value `shift`
// bits up. Images, their layouts and access, and the timeline semaphore of each are the frame's own: they are updated as it is read. `lock` and
// `unlock` (with `ctx`) keep its decoder off it meanwhile; `keep` holds the frame until the GPU is done with it.
struct VkDecodedFrame {
	VkImagePlane planes[3];
	int nplanes = 0;
	bool interleaved = false;
	int bytes = 1, shift = 0;
	VkImage images[3]{};
	int nimages = 0;
	VkImageLayout* layout[3]{};
	uint32_t* access[3]{};
	VkSemaphore sem[3]{};
	uint64_t* sem_value[3]{};
	void (*lock)(void* ctx) = nullptr;
	void (*unlock)(void* ctx) = nullptr;
	void* ctx = nullptr;
	const void* lock_domain = nullptr; // frames with the same one share their lock (taken once for both)
	int device = -1; // the Vulkan device (Vk::devices() index)
	std::shared_ptr<void> keep;
	// A frame in a buffer instead (the converter's output, vk_convert.cpp; nimages is 0 and the planes' img null): plane i at buf_offset[i], rows
	// of planes[i].w samples of `bytes` with no gaps. Readers wait for timeline semaphore `buf_sem` to reach `buf_value`; nothing in it changes
	// after that, so they signal nothing. `download` (a Picture*, see source.h) reads it into memory, for a reader that cannot take it in place.
	VkBuffer buffer = VK_NULL_HANDLE;
	VkDeviceSize buf_offset[3]{};
	VkSemaphore buf_sem = VK_NULL_HANDLE;
	uint64_t buf_value = 0;
	std::function<bool(void* picture)> download;
};

struct Buf {
	VkBuffer buf = VK_NULL_HANDLE;
	VkDeviceMemory mem = VK_NULL_HANDLE;
	void* map = nullptr; // set for host visible memory
	VkDeviceSize size = 0;
};

// A buffer whose memory another API can take in (CUDA's cuImportExternalMemory, for NVENC to read frames straight from VRAM): device local,
// in an allocation of its own (`alloc_size` bytes: what the importer is told). `win32` (an NT handle, Windows) or `fd` (elsewhere) is the
// memory's handle, which the caller owns: CloseHandle() it after the import (a successful CUDA import of an fd takes the fd over).
// Only buffers cross to the other API, never images: sharing Vulkan NV12 images with D3D11 or CUDA was seen to corrupt VRAM.
struct ExportedBuf {
	Buf b;
	VkDeviceSize alloc_size = 0;
	void* win32 = nullptr;
	int fd = -1;
};

struct Pipe {
	VkPipeline pipe = VK_NULL_HANDLE;
	VkDescriptorSet set = VK_NULL_HANDLE;
	VkDescriptorSet slot_sets[8] = {}; // with a buffer of Vk::alias() bound: one set per slot, with that slot's buffer in its place
	int nslot = 0;
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
	// Gives a pipeline()'s descriptor sets back to the pool (the pipeline itself is shared and stays); no work that binds it may be in flight.
	void release(Pipe& p);
	// Makes `placeholder`, wherever a pipeline made after this binds it, stand for per_slot[slot] in the work of that slot (begin(slot)): so each
	// frame in flight can read its own buffer, with no copy into one shared one.
	void alias(const Buf& placeholder, const std::vector<const Buf*>& per_slot);
	// Recording into the command buffer of a slot (whose earlier work must be done: wait()), then submitting it. Submissions to a device run in order
	// of submission on its one queue; a barrier() also orders the commands after it against everything submitted before (other slots, other users).
	void begin(int slot = 0);
	void bind(const Pipe& p);
	void push(const void* data, uint32_t size);
	void dispatch(uint32_t x, uint32_t y = 1);
	void copy(const Buf& src, const Buf& dst, VkDeviceSize size = 0); // size 0: the smaller of the two buffers
	// `rows` rows of `row_bytes` from src (at src_off, src_stride apart) to dst (at dst_off, with no gaps): one region where the rows have none either.
	void copy_rows(VkBuffer src, VkDeviceSize src_off, VkDeviceSize src_stride, VkBuffer dst, VkDeviceSize dst_off, VkDeviceSize row_bytes, uint32_t rows);
	// A timeline semaphore (starting at 0) that lives as long as this Vk.
	VkSemaphore timeline_semaphore();
	void copy_range(const Buf& src, VkDeviceSize src_off, const Buf& dst, VkDeviceSize dst_off, VkDeviceSize size);
	void barrier();
	// An image the device made (a decoded frame): moves it from `layout` (with writes `access` to finish first) to the layout copies read from,
	// and updates `layout` and `access` to that. Then copy_image() copies one of its planes into a buffer, row after row with no gaps.
	void image_to_transfer(VkImage img, VkImageLayout& layout, uint32_t& access);
	void copy_image(const VkImagePlane& plane, const Buf& dst, VkDeviceSize offset = 0);
	// The other way (the converter writing its frames into FFmpeg Vulkan frames): moves the image from `layout` (with `access` to finish first) to the
	// layout copies write to, updating both, then copy_to_image() copies rows of `row_texels` texels from `offset` in a buffer into one of its planes.
	void image_to_transfer_dst(VkImage img, VkImageLayout& layout, uint32_t& access);
	void copy_to_image(const Buf& src, VkDeviceSize offset, uint32_t row_texels, const VkImagePlane& plane);
	// The slot's next submit() waits for timeline semaphore `sem` to reach `value` first / sets it to `value` when done.
	void wait_timeline(int slot, VkSemaphore sem, uint64_t value);
	void signal_timeline(int slot, VkSemaphore sem, uint64_t value);
	// Whether the device came from a VkExternal provider (and so has timeline semaphores and can read its maker's images).
	bool external() const;
	static constexpr int kStamps = 12;
	void stamp(int slot, int which); // timestamp `which` (0: start, 1: end of the slot's work; up to kStamps - 1 for parts of it), see gpu_seconds()
	// Copies on a second queue (async_upload(): the device has one), so that copying a frame in overlaps the compute work on frames before it:
	// upload_begin() records into the slot's own command buffer for it, upload_submit() submits it, and the slot's next submit() waits for it.
	bool async_upload() const;
	void upload_begin(int slot);
	void upload_copy(const Buf& src, const Buf& dst, VkDeviceSize size = 0);
	void upload_copy_rows(VkBuffer src, VkDeviceSize src_off, VkDeviceSize src_stride, VkBuffer dst, VkDeviceSize dst_off, VkDeviceSize row_bytes, uint32_t rows);
	void upload_submit(int slot, bool for_second = false); // for_second: the slot's next submit2() waits for these copies too
	// ... and an image the device made, copied there too (as image_to_transfer(), copy_image(), wait_timeline() and signal_timeline(), for the
	// copies' command buffer and upload_submit()): the images must be usable by the copies' queue family as well (made for concurrent use).
	void upload_image_to_transfer(VkImage img, VkImageLayout& layout, uint32_t& access);
	void upload_copy_image(const VkImagePlane& plane, const Buf& dst);
	void upload_wait_timeline(int slot, VkSemaphore sem, uint64_t value);
	void upload_signal_timeline(int slot, VkSemaphore sem, uint64_t value);
	// A second compute queue (second_compute(): the device has one), for work that needs nothing from the main queue's and can run beside it:
	// begin2() records the slot's work for it (with bind/push/dispatch/copy/barrier as usual; stamp() does nothing there), submit2() submits it, and
	// wait() waits for it as well. Work on it runs in submission order only against itself.
	bool second_compute() const;
	void begin2(int slot);
	void submit2(int slot);
	void submit(int slot = 0);
	void wait(int slot = 0); // until the slot's last submission is done
	bool done(int slot = 0); // whether the slot's last submission is done, without waiting
	// GPU time between two of the slot's stamps (by default start and end) in its last finished submission; -1 if the device has no timestamps or
	// the submission did not write both. Safe to call from another thread than the one that records, for a slot that is not being recorded.
	double gpu_seconds(int slot, int from = 0, int to = 1);
	// VES_SCORE_TIMING=2: a timestamp after every dispatch on the main queue (up to kPassStamps per submission). The GPU time of each dispatch of
	// the slot's last finished submission, in order, with its shader's name (from the end of the one before; the first from the start).
	static constexpr int kPassStamps = 1024;
	bool pass_timing() const;
	void pass_seconds(int slot, std::vector<std::pair<const char*, double>>& out);
	// A buffer of `size` bytes for exporting (see ExportedBuf), freed with this Vk like the others. False with `why` when the device cannot
	// export memory (made without the extension, or the driver refuses).
	bool exportable_buffer(VkDeviceSize size, ExportedBuf& out, std::string& why);
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
