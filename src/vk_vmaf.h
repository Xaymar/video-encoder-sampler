#pragma once
// VMAF scoring on Vulkan compute (see vk_vmaf.cpp), for 8 and 10 bit frames: the VMAF v0 features (VIF, ADM, motion; luma only) and the VMAF v1
// ones (adm3, motion3, CAMBI on the luma plane, SpEED on the chroma planes).
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

// What is scored: w x h at `depth` bits, chroma 420, 422 or 444 (400: no chroma planes). The encode was enc_w x enc_h at enc_depth bits, which
// is what CAMBI looks at (vmaf.cpp's cambi_encode() works out the values libvmaf is given; the GPU scorer is given the same).
struct GpuVmafFrame {
	int w = 0, h = 0, depth = 8, chroma = 420;
	int enc_w = 0, enc_h = 0, enc_depth = 8;
};

class GpuVmaf {
public:
	// The Vulkan device to score on (an index into Vk::devices()), -1 (and `why`) if there is none. `spec` is vmaf.gpu: "" or "auto" for the fastest
	// GPU that can score (vulkan_devices_fastest_first(); "own" for the GPU that encodes (GPU `encoder_gpu` of vendor
	// `encoder_vendor`, see gpu.h), else the fastest; a number for that Vulkan device; anything else for the first device whose name contains it (any case).
	static int pick_device(const std::string& spec, unsigned encoder_vendor, int encoder_gpu, std::string& why);
	// Whether the GPU scorer can score these frames with these model files on Vulkan device `device`; `why` says what is in the way.
	// Several models are scored in one pass: what they compute alike (motion, CAMBI, the ADM wavelets, the SpEED planes) is computed once.
	static bool supported(const std::vector<std::string>& model_paths, int device, const GpuVmafFrame& f, std::string& why);
	// A scorer for one run; nullptr (and `why`) if it cannot be had (no suitable GPU, not enough video memory, an unsupported model).
	static std::unique_ptr<GpuVmaf> create(const std::vector<std::string>& model_paths, int device, const GpuVmafFrame& f, std::string& why);
	~GpuVmaf();
	// The next pair of original and decoded frames: planes Y, U, V (samples of 1 byte, or 2 bytes above 8 bit; rows `stride` bytes apart). The chroma
	// planes are only read when the model has a chroma feature.
	bool add(const uint8_t* const ref[3], const size_t ref_stride[3], const uint8_t* const dis[3], const size_t dis_stride[3]);
	// For each model (in the order given), the VMAF score of every pair added; and, if asked for, every frame's features as libvmaf names them.
	bool finish(std::vector<std::vector<double>>& scores, std::vector<std::vector<std::map<std::string, double>>>* features = nullptr);

private:
	GpuVmaf();
	struct Impl;
	Impl* impl;
};
