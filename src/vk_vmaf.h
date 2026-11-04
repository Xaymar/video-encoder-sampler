#pragma once
// VMAF scoring on Vulkan compute (see vk_vmaf.cpp). 8 and 10 bit luma only (VMAF looks at nothing else); the scores match libvmaf's within 1e-4.
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

class GpuVmaf {
public:
	// The Vulkan device to score on (an index into Vk::devices()), -1 (and `why`) if there is none. `spec` is vmaf.gpu: "" or "auto" for the GPU that
	// encodes (GPU `encoder_gpu` of vendor `encoder_vendor`, see gpu.h; vendor 0 for a CPU encoder) where it can score, else the first discrete GPU;
	// a number for that Vulkan device; anything else for the first device whose name contains it (any case).
	static int pick_device(const std::string& spec, unsigned encoder_vendor, int encoder_gpu, std::string& why);
	// Whether the GPU scorer can score `w` x `h` frames of `depth` bits with this model file on Vulkan device `device`; `why` says what is in the way.
	static bool supported(const std::string& model_path, int device, int w, int h, int depth, std::string& why);
	// A scorer for one run; nullptr (and `why`) if it cannot be had (no suitable GPU, not enough video memory, an unsupported model).
	static std::unique_ptr<GpuVmaf> create(const std::string& model_path, int device, int w, int h, int depth, std::string& why);
	~GpuVmaf();
	// The next pair of original and decoded luma planes (w x h samples of 1 byte, or 2 bytes above 8 bit; rows `stride` bytes apart).
	bool add(const uint8_t* ref, size_t ref_stride, const uint8_t* dis, size_t dis_stride);
	// The VMAF score of every pair added.
	bool finish(std::vector<double>& scores);

private:
	GpuVmaf();
	struct Impl;
	Impl* impl;
};
