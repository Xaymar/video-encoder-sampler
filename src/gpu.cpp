#include "gpu.h"

#include <windows.h>
#include <dxgi.h>
#include <wrl/client.h>

std::vector<GpuInfo> list_gpus(unsigned vendor) {
	std::vector<GpuInfo> out;
	Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
	if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return out;
	Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
	for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; i++, adapter.Reset()) {
		DXGI_ADAPTER_DESC1 d;
		adapter->GetDesc1(&d);
		if (d.VendorId != vendor || (d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
		LARGE_INTEGER umd{};
		adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd);
		char name[256];
		WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, name, sizeof name, nullptr, nullptr);
		char driver[64];
		snprintf(driver, sizeof driver, "%u.%u.%u.%u", HIWORD(umd.HighPart), LOWORD(umd.HighPart), HIWORD(umd.LowPart), LOWORD(umd.LowPart));
		GpuInfo g;
		g.name = name;
		g.driver = driver;
		if (vendor == VENDOR_NVIDIA) { // 32.0.16.1742 is driver 617.42: the last digit of the third part, then the fourth
			char marketing[32];
			snprintf(marketing, sizeof marketing, "%u.%02u", (HIWORD(umd.LowPart) % 10) * 100 + LOWORD(umd.LowPart) / 100, LOWORD(umd.LowPart) % 100);
			g.driver = std::string(marketing) + " (" + driver + ")";
		}
		g.luid = (uint64_t)d.AdapterLuid.LowPart | ((uint64_t)(uint32_t)d.AdapterLuid.HighPart << 32);
		out.push_back(std::move(g));
	}
	return out;
}

std::string gpu_description(unsigned vendor, int index) {
	auto all = list_gpus(vendor);
	return index >= 0 && index < (int)all.size() ? all[index].describe() : "none";
}

unsigned encoder_vendor(const std::string& encoder) { return encoder == "nvenc" ? VENDOR_NVIDIA : encoder == "amf" ? VENDOR_AMD : 0; }
