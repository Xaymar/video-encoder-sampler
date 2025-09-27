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
		g.vendor_id = d.VendorId; g.device_id = d.DeviceId; g.sub_system_id = d.SubSysId; g.revision = d.Revision;
		g.luid = (uint64_t)d.AdapterLuid.LowPart | ((uint64_t)(uint32_t)d.AdapterLuid.HighPart << 32);
		out.push_back(std::move(g));
	}
	return out;
}

static std::string json_string(const std::string& s) {
	std::string o = "\"";
	for (unsigned char c : s) {
		if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
		else if (c >= 0x20) o += (char)c;
	}
	return o + "\"";
}

std::string gpu_json(unsigned vendor, int index) {
	auto all = list_gpus(vendor);
	if (index < 0 || index >= (int)all.size()) return "null";
	const GpuInfo& g = all[index];
	char ids[160];
	snprintf(ids, sizeof ids, "\"vendor_id\":\"%04x\",\"device_id\":\"%04x\",\"sub_system_id\":\"%08x\",\"revision\":\"%02x\"", g.vendor_id, g.device_id, g.sub_system_id, g.revision);
	return "{\"name\":" + json_string(g.name) + ",\"driver\":" + json_string(g.driver) + "," + ids + "}";
}

std::string gpu_description(unsigned vendor, int index) {
	auto all = list_gpus(vendor);
	return index >= 0 && index < (int)all.size() ? all[index].describe() : "none";
}

unsigned encoder_vendor(const std::string& encoder) { return encoder == "nvenc" ? VENDOR_NVIDIA : encoder == "amf" ? VENDOR_AMD : encoder == "qsv" ? VENDOR_INTEL : 0; }
