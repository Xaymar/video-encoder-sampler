#pragma once
// Graphics adapters as Windows sees them (DXGI), the same list the GPU encoders and the result output use.
#include <cstdint>
#include <string>
#include <vector>

constexpr unsigned VENDOR_NVIDIA = 0x10DE;
constexpr unsigned VENDOR_AMD = 0x1002;
constexpr unsigned VENDOR_INTEL = 0x8086;

struct GpuInfo {
	std::string name;
	std::string driver; // "a.b.c.d" of the user mode driver
	uint64_t luid = 0;  // adapter LUID (low part | high part << 32), identifies the same GPU to CUDA and Vulkan
	std::string describe() const { return name + " (driver " + driver + ")"; }
};

// All adapters of a vendor in DXGI order; the n-th one is what `gpu=n` means for that vendor's encoder.
std::vector<GpuInfo> list_gpus(unsigned vendor);

// "<name> (driver <version>)" of the n-th adapter of a vendor, or "none".
std::string gpu_description(unsigned vendor, int index);

// The vendor behind an encoder name, 0 for encoders that do not run on a GPU.
unsigned encoder_vendor(const std::string& encoder);
