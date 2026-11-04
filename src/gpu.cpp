#include "gpu.h"

#include <cstdio>
#include <map>
#include <mutex>

#ifdef _WIN32
#include <windows.h>
#include <dxgi.h>
#include <wrl/client.h>

static std::vector<GpuInfo> enumerate_gpus(unsigned vendor) {
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

#else
#include <algorithm>
#include <filesystem>
#include <fstream>

// Linux has no adapter list API without a graphics library; the kernel's DRM nodes in sysfs carry the PCI ids and the driver.
static std::string sysfs_text(const std::filesystem::path& p) {
	std::ifstream f(p);
	std::string s;
	std::getline(f, s);
	return s;
}

static std::vector<GpuInfo> enumerate_gpus(unsigned vendor) {
	namespace fs = std::filesystem;
	std::vector<GpuInfo> out;
	std::vector<fs::path> cards;
	std::error_code ec;
	for (auto& e : fs::directory_iterator("/sys/class/drm", ec)) {
		const std::string n = e.path().filename().string();
		if (n.rfind("card", 0) == 0 && n.find('-') == std::string::npos) cards.push_back(e.path() / "device");
	}
	std::sort(cards.begin(), cards.end());
	for (auto& dev : cards) {
		GpuInfo g;
		g.vendor_id = (uint32_t)strtoul(sysfs_text(dev / "vendor").c_str(), nullptr, 16);
		if (g.vendor_id != vendor) continue;
		g.device_id = (uint32_t)strtoul(sysfs_text(dev / "device").c_str(), nullptr, 16);
		g.sub_system_id = (uint32_t)strtoul(sysfs_text(dev / "subsystem_device").c_str(), nullptr, 16) | ((uint32_t)strtoul(sysfs_text(dev / "subsystem_vendor").c_str(), nullptr, 16) << 16);
		g.revision = (uint32_t)strtoul(sysfs_text(dev / "revision").c_str(), nullptr, 16);
		char name[64];
		snprintf(name, sizeof name, "GPU %04x:%04x", g.vendor_id, g.device_id); // sysfs has no marketing name; the ids identify the model
		g.name = name;
		const std::string kernel = fs::path(fs::read_symlink(dev / "driver", ec)).filename().string();
		g.driver = kernel.empty() ? "unknown" : kernel;
		if (vendor == VENDOR_NVIDIA) {
			const std::string version = sysfs_text("/sys/module/nvidia/version");
			if (!version.empty()) g.driver = version;
		}
		out.push_back(std::move(g));
	}
	return out;
}
#endif

std::vector<GpuInfo> list_gpus(unsigned vendor) {
	static std::mutex m;
	static std::map<unsigned, std::vector<GpuInfo>> known;
	std::lock_guard<std::mutex> lock(m);
	auto it = known.find(vendor);
	if (it == known.end()) it = known.emplace(vendor, enumerate_gpus(vendor)).first;
	return it->second;
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

#ifdef _WIN32
int dxgi_adapter_index(unsigned vendor, int index) {
	const auto gpus = list_gpus(vendor);
	if (index < 0 || index >= (int)gpus.size()) return -1;
	Microsoft::WRL::ComPtr<IDXGIFactory1> factory;
	if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory)))) return -1;
	Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
	for (UINT i = 0; factory->EnumAdapters1(i, &adapter) == S_OK; i++, adapter.Reset()) {
		DXGI_ADAPTER_DESC1 d;
		adapter->GetDesc1(&d);
		if (((uint64_t)d.AdapterLuid.LowPart | ((uint64_t)(uint32_t)d.AdapterLuid.HighPart << 32)) == gpus[(size_t)index].luid) return (int)i;
	}
	return -1;
}
std::string render_node(unsigned, int) { return {}; }
#else
int dxgi_adapter_index(unsigned, int) { return -1; }
std::string render_node(unsigned vendor, int index) {
	namespace fs = std::filesystem;
	std::vector<fs::path> cards;
	std::error_code ec;
	for (auto& e : fs::directory_iterator("/sys/class/drm", ec)) {
		const std::string n = e.path().filename().string();
		if (n.rfind("card", 0) == 0 && n.find('-') == std::string::npos) cards.push_back(e.path() / "device");
	}
	std::sort(cards.begin(), cards.end()); // the order enumerate_gpus() lists them in
	int seen = 0;
	for (auto& dev : cards) {
		if ((unsigned)strtoul(sysfs_text(dev / "vendor").c_str(), nullptr, 16) != vendor) continue;
		if (seen++ != index) continue;
		for (auto& e : fs::directory_iterator(dev / "drm", ec)) {
			const std::string n = e.path().filename().string();
			if (n.rfind("renderD", 0) == 0) return "/dev/dri/" + n;
		}
		return {};
	}
	return {};
}
#endif
