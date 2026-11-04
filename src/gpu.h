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
	uint32_t vendor_id = 0, device_id = 0, sub_system_id = 0, revision = 0; // PCI identity: what tells GPU models, and so encoder generations, apart
	std::string describe() const { return name + " (driver " + driver + ")"; }
};

// All adapters of a vendor in DXGI order; the n-th one is what `gpu=n` means for that vendor's encoder.
std::vector<GpuInfo> list_gpus(unsigned vendor);

// The n-th adapter of a vendor as a JSON object (name, driver and PCI ids), or "null".
std::string gpu_json(unsigned vendor, int index);

// "<name> (driver <version>)" of the n-th adapter of a vendor, or "none".
std::string gpu_description(unsigned vendor, int index);

// The vendor behind an encoder name, 0 for encoders that do not run on a GPU.
unsigned encoder_vendor(const std::string& encoder);

// Where the n-th GPU of a vendor is to other APIs, for decoding on that very GPU: its index among all DXGI adapters (what FFmpeg's d3d11va and d3d12va
// devices take; Windows, -1 elsewhere or when it is not found), and its DRM render node (/dev/dri/renderD128, what vaapi takes; Linux, empty elsewhere).
int dxgi_adapter_index(unsigned vendor, int index);
std::string render_node(unsigned vendor, int index);
