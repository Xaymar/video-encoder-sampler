// Checks an NV12 texture shared between D3D11 and Vulkan on the same GPU: each API writes a known picture into it, the other reads it back,
// and guard resources made right before and after the shared texture are checked for damage.
//
// --dir d3d11 (default): D3D11 makes the NV12 texture (D3D11_RESOURCE_MISC_SHARED | SHARED_NTHANDLE), Vulkan imports it as
//   VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, optimal tiling, with VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT and a dedicated allocation.
// --dir vk: Vulkan makes the NV12 image with exportable dedicated memory (VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT), D3D11 opens the
//   NT handle with ID3D11Device1::OpenSharedResource1.
// --format r8: control with the same bytes in a plain single-plane R8 texture (width x height*3/2) instead of NV12.
//
// Steps per size (each step waits for the GPU to finish before the next one; nothing runs at the same time):
//   1. D3D11 writes picture A (staging texture + CopyResource); Vulkan copies both planes out (vkCmdCopyImageToBuffer, PLANE_0 / PLANE_1).
//   2. Vulkan writes picture B (vkCmdCopyBufferToImage per plane); D3D11 reads it back (CopyResource to a staging texture, Map).
//   3. Vulkan reads picture B back itself.
//   4. The guards (Vulkan device-local buffers, D3D11 R8 textures, each one NV12 picture large, filled with a known value) are checked.
// Every mismatch is counted per plane; where the expected first row of a plane turns up somewhere else, the program prints where.
// Exit code 0: every check passed; 1: mismatches; 2: setup failed.
//
// Measured on Irkela (RTX 4070 Laptop GPU, driver 621.24, Windows 10 22H2 19045.6466, Vulkan loader 1.4.363), 2026-11-20:
//   --dir d3d11, NV12: plane 0 correct both ways; plane 1 (UV) is at a different place for Vulkan than for D3D11, both ways (~99.6% of its
//   bytes wrong). What Vulkan sees as UV row 0 is D3D11's UV row 32 at 1280x720, x 1024 row 64 at 1920x1080, x 2048 row 0 at 3840x2160.
//   Each API reads its own writes back correctly. Every run damages the first Vulkan buffer made after the shared texture (from byte 0:
//   6656 words at 720p, 29824 at 1080p, 25088 at 2160p), so Vulkan's plane 1 writes go past the end of the D3D11 allocation.
//   The third 1920x1080 iteration of a run lost the device (VK_ERROR_DEVICE_LOST, nvlddmkm event 153 "Error occurred on GPUID: 100").
//   --old-layout undefined: the same. --format r8 (same bytes, R8 texture): everything correct, no guard damaged.
//   --dir vk: OpenSharedResource1 returns E_INVALIDARG for NV12 and R8 alike (OPAQUE_WIN32 is not meant for D3D11), so that direction is untested.
//
// Uses only D3D11/DXGI and the Vulkan loader (vulkan-1.dll, loaded at run time).
//
// Build (Windows, clang; the Vulkan headers' include folder is the only dependency, https://github.com/KhronosGroup/Vulkan-Headers or the
// Vulkan SDK; NOMINMAX because vulkan_win32.h includes windows.h):
//   clang++ -std=c++17 -O2 -DNOMINMAX -I <Vulkan-Headers>/include nv12_vulkan_d3d11_sharing.cpp -o nv12_vulkan_d3d11_sharing.exe -ld3d11 -ldxgi
//   (MinGW: x86_64-w64-mingw32-g++ the same way, plus -static)
// Run:
//   nv12_vulkan_d3d11_sharing.exe [--dir d3d11|vk] [--format nv12|r8] [--size WxH ...] [--iterations n] [--adapter n]
//                                 [--old-layout general|undefined] [--guards n]
//   (defaults: --dir d3d11 --format nv12 --size 1280x720 --size 1920x1080 --size 3840x2160 --iterations 1, first NVIDIA adapter, --guards 4)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <d3d11_1.h>
#include <dxgi1_2.h>
#define VK_NO_PROTOTYPES
#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define LOG(...) do { printf(__VA_ARGS__); printf("\n"); fflush(stdout); } while (0)
#define VKCHECK(call) do { VkResult r_ = (call); if (r_ != VK_SUCCESS) { LOG("%s failed: VkResult %d (line %d)", #call, (int)r_, __LINE__); return false; } } while (0)
#define HRCHECK(call) do { HRESULT h_ = (call); if (FAILED(h_)) { LOG("%s failed: HRESULT 0x%08lx (line %d)", #call, (unsigned long)h_, __LINE__); return false; } } while (0)

template <class T> struct Com {
	T* p = nullptr;
	Com() = default;
	Com(const Com&) = delete;
	Com& operator=(const Com&) = delete;
	~Com() { reset(); }
	void reset() { if (p) p->Release(); p = nullptr; }
	T** put() { reset(); return &p; }
	T* operator->() const { return p; }
	operator T*() const { return p; }
};

// ---- Vulkan entry points -------------------------------------------------------------------------------------------------------------------
#define VK_GLOBAL_FNS(X) X(vkCreateInstance) X(vkEnumerateInstanceVersion)
#define VK_INSTANCE_FNS(X)                                                                                                                         \
	X(vkDestroyInstance) X(vkEnumeratePhysicalDevices) X(vkGetPhysicalDeviceProperties2) X(vkGetPhysicalDeviceQueueFamilyProperties)            \
	X(vkGetPhysicalDeviceMemoryProperties) X(vkGetPhysicalDeviceImageFormatProperties2) X(vkGetPhysicalDeviceFormatProperties) X(vkCreateDevice) \
	X(vkGetDeviceProcAddr) X(vkEnumerateDeviceExtensionProperties)
#define VK_DEVICE_FNS(X)                                                                                                                           \
	X(vkDestroyDevice) X(vkGetDeviceQueue) X(vkCreateImage) X(vkDestroyImage) X(vkGetImageMemoryRequirements2) X(vkAllocateMemory)              \
	X(vkFreeMemory) X(vkBindImageMemory) X(vkCreateBuffer) X(vkDestroyBuffer) X(vkGetBufferMemoryRequirements) X(vkBindBufferMemory)            \
	X(vkMapMemory) X(vkUnmapMemory) X(vkCreateCommandPool) X(vkDestroyCommandPool) X(vkAllocateCommandBuffers) X(vkBeginCommandBuffer)         \
	X(vkEndCommandBuffer) X(vkResetCommandBuffer) X(vkCmdPipelineBarrier) X(vkCmdCopyImageToBuffer) X(vkCmdCopyBufferToImage)                  \
	X(vkCmdFillBuffer) X(vkCmdCopyBuffer) X(vkQueueSubmit) X(vkQueueWaitIdle) X(vkDeviceWaitIdle) X(vkGetMemoryWin32HandleKHR)                 \
	X(vkGetMemoryWin32HandlePropertiesKHR)
#define DECLARE_FN(n) static PFN_##n n;
VK_GLOBAL_FNS(DECLARE_FN)
VK_INSTANCE_FNS(DECLARE_FN)
VK_DEVICE_FNS(DECLARE_FN)
static PFN_vkGetInstanceProcAddr vkGetInstanceProcAddr_;

// ---- Options and the picture ---------------------------------------------------------------------------------------------------------------
struct Options {
	bool vk_exports = false; // --dir vk
	bool r8 = false;         // --format r8
	std::vector<std::pair<int, int>> sizes;
	int iterations = 1;
	int adapter = -1; // -1: first NVIDIA adapter
	bool old_layout_undefined = false;
	int guards = 4;
};

struct Plane {
	int row_bytes; // bytes per row
	int rows;
	VkImageAspectFlagBits aspect;
	int vk_width; // texels per row in Vulkan
};

static std::vector<Plane> planes_of(const Options& op, int w, int h) {
	if (op.r8) return {{w, h * 3 / 2, VK_IMAGE_ASPECT_COLOR_BIT, w}};
	return {{w, h, VK_IMAGE_ASPECT_PLANE_0_BIT, w}, {w, h / 2, VK_IMAGE_ASPECT_PLANE_1_BIT, w / 2}};
}

// One byte of a picture: a hash of (seed, plane, x, y), so a row found at the wrong place can be told apart from every other row.
static uint8_t pattern(uint32_t seed, int plane, int x, int y) {
	uint32_t v = (uint32_t)x * 0x9E3779B1u ^ (uint32_t)y * 0x85EBCA77u ^ seed * 0xC2B2AE3Du ^ (uint32_t)plane * 0x27D4EB2Fu;
	v ^= v >> 15;
	v *= 0x2C1B3C6Du;
	v ^= v >> 13;
	return (uint8_t)v;
}

// The picture tightly packed: plane 0 rows, then plane 1 rows (the NV12 layout without padding).
static std::vector<uint8_t> make_picture(const std::vector<Plane>& planes, uint32_t seed) {
	std::vector<uint8_t> out;
	for (size_t p = 0; p < planes.size(); p++)
		for (int y = 0; y < planes[p].rows; y++)
			for (int x = 0; x < planes[p].row_bytes; x++) out.push_back(pattern(seed, (int)p, x, y));
	return out;
}

static size_t plane_offset(const std::vector<Plane>& planes, size_t p) {
	size_t o = 0;
	for (size_t i = 0; i < p; i++) o += (size_t)planes[i].row_bytes * planes[i].rows;
	return o;
}

static long long find_bytes(const uint8_t* hay, size_t n, const uint8_t* needle, size_t k) {
	if (n < k) return -1;
	for (size_t i = 0; i + k <= n; i++)
		if (hay[i] == needle[0] && memcmp(hay + i, needle, k) == 0) return (long long)i;
	return -1;
}

// Where offset o of a tightly packed picture lies: "plane p x,y".
static std::string where(const std::vector<Plane>& planes, long long o) {
	char b[96];
	for (size_t p = 0; p < planes.size(); p++) {
		const long long n = (long long)planes[p].row_bytes * planes[p].rows;
		if (o < n) {
			snprintf(b, sizeof b, "plane %zu x %lld y %lld", p, o % planes[p].row_bytes, o / planes[p].row_bytes);
			return b;
		}
		o -= n;
	}
	snprintf(b, sizeof b, "past the end (+%lld)", o);
	return b;
}

// Compares a tightly packed picture with the expected one; prints a summary per plane. Returns the number of wrong bytes.
static size_t compare(const char* what, const std::vector<Plane>& planes, const std::vector<uint8_t>& got, const std::vector<uint8_t>& want) {
	size_t total = 0;
	for (size_t p = 0; p < planes.size(); p++) {
		const size_t base = plane_offset(planes, p), rb = planes[p].row_bytes;
		size_t wrong = 0, rows_any = 0, rows_all = 0;
		long long first = -1;
		for (int y = 0; y < planes[p].rows; y++) {
			size_t row_wrong = 0;
			for (size_t x = 0; x < rb; x++) {
				const size_t i = base + y * rb + x;
				if (got[i] != want[i]) {
					if (first < 0) first = (long long)i;
					row_wrong++;
				}
			}
			wrong += row_wrong;
			rows_any += row_wrong != 0;
			rows_all += row_wrong == rb;
		}
		total += wrong;
		if (!wrong) {
			LOG("  %s plane %zu: OK", what, p);
			continue;
		}
		LOG("  %s plane %zu: %zu of %zu bytes wrong, %zu of %d rows touched, %zu rows entirely wrong; first at %s (expected 0x%02x, got 0x%02x)", what, p,
		    wrong, rb * planes[p].rows, rows_any, planes[p].rows, rows_all, where(planes, first).c_str(), want[first], got[first]);
		// Where did the start of this plane's expected first wrong row end up, and what is at its place instead?
		const size_t row_start = base + (first - base) / rb * rb;
		const size_t k = std::min<size_t>(32, rb);
		const long long at = find_bytes(got.data(), got.size(), want.data() + row_start, k);
		if (at >= 0) LOG("    expected %s is found at %s", where(planes, (long long)row_start).c_str(), where(planes, at).c_str());
		else LOG("    expected %s is not found anywhere in what was read", where(planes, (long long)row_start).c_str());
		const long long from = find_bytes(want.data(), want.size(), got.data() + row_start, k);
		if (from >= 0) LOG("    what was read there belongs to %s", where(planes, from).c_str());
	}
	return total;
}

// ---- D3D11 ---------------------------------------------------------------------------------------------------------------------------------
struct D3D {
	Com<IDXGIAdapter1> adapter;
	Com<ID3D11Device> dev;
	Com<ID3D11Device1> dev1;
	Com<ID3D11DeviceContext> ctx;
	DXGI_ADAPTER_DESC1 desc{};
};

static bool d3d_init(D3D& d, int want) {
	Com<IDXGIFactory1> f;
	HRCHECK(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)f.put()));
	for (UINT i = 0;; i++) {
		Com<IDXGIAdapter1> a;
		if (f->EnumAdapters1(i, a.put()) == DXGI_ERROR_NOT_FOUND) break;
		DXGI_ADAPTER_DESC1 desc;
		a->GetDesc1(&desc);
		LOG("DXGI adapter %u: %ls (vendor 0x%04x, LUID %08lx:%08lx)", i, desc.Description, desc.VendorId, (unsigned long)desc.AdapterLuid.HighPart,
		    (unsigned long)desc.AdapterLuid.LowPart);
		if (d.adapter.p) continue;
		if ((want < 0 && desc.VendorId == 0x10de) || want == (int)i) {
			d.adapter.p = a.p;
			a.p = nullptr;
			d.desc = desc;
		}
	}
	if (!d.adapter.p) {
		LOG("no matching DXGI adapter (use --adapter n)");
		return false;
	}
	LARGE_INTEGER umd{};
	if (SUCCEEDED(d.adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &umd)))
		LOG("using %ls, UMD driver %u.%u.%u.%u", d.desc.Description, HIWORD(umd.HighPart), LOWORD(umd.HighPart), HIWORD(umd.LowPart), LOWORD(umd.LowPart));
	const D3D_FEATURE_LEVEL fl[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
	HRCHECK(D3D11CreateDevice(d.adapter, D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, fl, 2, D3D11_SDK_VERSION, d.dev.put(), nullptr, d.ctx.put()));
	HRCHECK(d.dev->QueryInterface(__uuidof(ID3D11Device1), (void**)d.dev1.put()));
	return true;
}

static void d3d_wait(D3D& d) {
	D3D11_QUERY_DESC qd{D3D11_QUERY_EVENT, 0};
	Com<ID3D11Query> q;
	if (FAILED(d.dev->CreateQuery(&qd, q.put()))) return;
	d.ctx->End(q);
	d.ctx->Flush();
	BOOL done = FALSE;
	while (d.ctx->GetData(q, &done, sizeof done, 0) != S_OK) Sleep(0);
}

static D3D11_TEXTURE2D_DESC tex_desc(const Options& op, int w, int h) {
	D3D11_TEXTURE2D_DESC t{};
	t.Width = w;
	t.Height = op.r8 ? h * 3 / 2 : h;
	t.MipLevels = 1;
	t.ArraySize = 1;
	t.Format = op.r8 ? DXGI_FORMAT_R8_UNORM : DXGI_FORMAT_NV12;
	t.SampleDesc.Count = 1;
	t.Usage = D3D11_USAGE_DEFAULT;
	t.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
	return t;
}

static bool make_staging(D3D& d, D3D11_TEXTURE2D_DESC t, Com<ID3D11Texture2D>& out) {
	t.Usage = D3D11_USAGE_STAGING;
	t.BindFlags = 0;
	t.MiscFlags = 0;
	t.CPUAccessFlags = D3D11_CPU_ACCESS_READ | D3D11_CPU_ACCESS_WRITE;
	HRCHECK(d.dev->CreateTexture2D(&t, nullptr, out.put()));
	return true;
}

// Copies a tightly packed picture into tex (whose rows are the planes' rows one after another, as D3D11 maps NV12 and R8 alike).
static bool d3d_write(D3D& d, ID3D11Texture2D* tex, const std::vector<Plane>& planes, const std::vector<uint8_t>& pic) {
	D3D11_TEXTURE2D_DESC t;
	tex->GetDesc(&t);
	Com<ID3D11Texture2D> st;
	if (!make_staging(d, t, st)) return false;
	D3D11_MAPPED_SUBRESOURCE m;
	HRCHECK(d.ctx->Map(st, 0, D3D11_MAP_WRITE, 0, &m));
	size_t row = 0;
	for (size_t p = 0; p < planes.size(); p++)
		for (int y = 0; y < planes[p].rows; y++, row++)
			memcpy((uint8_t*)m.pData + row * m.RowPitch, pic.data() + plane_offset(planes, p) + (size_t)y * planes[p].row_bytes, planes[p].row_bytes);
	d.ctx->Unmap(st, 0);
	d.ctx->CopyResource(tex, st);
	d3d_wait(d);
	return true;
}

static bool d3d_read(D3D& d, ID3D11Texture2D* tex, const std::vector<Plane>& planes, std::vector<uint8_t>& pic) {
	D3D11_TEXTURE2D_DESC t;
	tex->GetDesc(&t);
	Com<ID3D11Texture2D> st;
	if (!make_staging(d, t, st)) return false;
	d.ctx->CopyResource(st, tex);
	d3d_wait(d);
	D3D11_MAPPED_SUBRESOURCE m;
	HRCHECK(d.ctx->Map(st, 0, D3D11_MAP_READ, 0, &m));
	pic.assign(plane_offset(planes, planes.size()), 0);
	size_t row = 0;
	for (size_t p = 0; p < planes.size(); p++)
		for (int y = 0; y < planes[p].rows; y++, row++)
			memcpy(pic.data() + plane_offset(planes, p) + (size_t)y * planes[p].row_bytes, (uint8_t*)m.pData + row * m.RowPitch, planes[p].row_bytes);
	d.ctx->Unmap(st, 0);
	return true;
}

// ---- Vulkan --------------------------------------------------------------------------------------------------------------------------------
struct VK {
	HMODULE lib = nullptr;
	VkInstance inst = VK_NULL_HANDLE;
	VkPhysicalDevice phys = VK_NULL_HANDLE;
	VkDevice dev = VK_NULL_HANDLE;
	uint32_t family = 0;
	VkQueue queue = VK_NULL_HANDLE;
	VkCommandPool pool = VK_NULL_HANDLE;
	VkCommandBuffer cmd = VK_NULL_HANDLE;
	VkPhysicalDeviceMemoryProperties mem{};
};

static int memory_type(const VK& v, uint32_t bits, VkMemoryPropertyFlags want) {
	for (uint32_t i = 0; i < v.mem.memoryTypeCount; i++)
		if ((bits & (1u << i)) && (v.mem.memoryTypes[i].propertyFlags & want) == want) return (int)i;
	return -1;
}

static bool vk_init(VK& v, const LUID& luid) {
	v.lib = LoadLibraryA("vulkan-1.dll");
	if (!v.lib) {
		LOG("vulkan-1.dll not found");
		return false;
	}
	vkGetInstanceProcAddr_ = (PFN_vkGetInstanceProcAddr)(void*)GetProcAddress(v.lib, "vkGetInstanceProcAddr");
#define LOAD_GLOBAL(n) n = (PFN_##n)vkGetInstanceProcAddr_(VK_NULL_HANDLE, #n);
	VK_GLOBAL_FNS(LOAD_GLOBAL)
	uint32_t api = VK_API_VERSION_1_0;
	if (vkEnumerateInstanceVersion) vkEnumerateInstanceVersion(&api);
	LOG("Vulkan loader %u.%u.%u", VK_API_VERSION_MAJOR(api), VK_API_VERSION_MINOR(api), VK_API_VERSION_PATCH(api));
	VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
	ai.pApplicationName = "nv12_vulkan_d3d11_sharing";
	ai.apiVersion = VK_API_VERSION_1_2;
	VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
	ici.pApplicationInfo = &ai;
	VKCHECK(vkCreateInstance(&ici, nullptr, &v.inst));
#define LOAD_INSTANCE(n) n = (PFN_##n)vkGetInstanceProcAddr_(v.inst, #n);
	VK_INSTANCE_FNS(LOAD_INSTANCE)

	uint32_t n = 0;
	vkEnumeratePhysicalDevices(v.inst, &n, nullptr);
	std::vector<VkPhysicalDevice> devs(n);
	vkEnumeratePhysicalDevices(v.inst, &n, devs.data());
	for (VkPhysicalDevice pd : devs) {
		VkPhysicalDeviceDriverProperties drv{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES};
		VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES, &drv};
		VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, &id};
		vkGetPhysicalDeviceProperties2(pd, &p2);
		const bool match = id.deviceLUIDValid && memcmp(id.deviceLUID, &luid, VK_LUID_SIZE) == 0;
		LOG("Vulkan device: %s, %s %s, API %u.%u.%u%s", p2.properties.deviceName, drv.driverName, drv.driverInfo, VK_API_VERSION_MAJOR(p2.properties.apiVersion),
		    VK_API_VERSION_MINOR(p2.properties.apiVersion), VK_API_VERSION_PATCH(p2.properties.apiVersion), match ? "  <- same LUID as the D3D11 adapter" : "");
		if (match && !v.phys) v.phys = pd;
	}
	if (!v.phys) {
		LOG("no Vulkan device with the D3D11 adapter's LUID");
		return false;
	}
	vkGetPhysicalDeviceMemoryProperties(v.phys, &v.mem);

	vkGetPhysicalDeviceQueueFamilyProperties(v.phys, &n, nullptr);
	std::vector<VkQueueFamilyProperties> qf(n);
	vkGetPhysicalDeviceQueueFamilyProperties(v.phys, &n, qf.data());
	for (v.family = 0; v.family < n && !(qf[v.family].queueFlags & VK_QUEUE_GRAPHICS_BIT); v.family++) {}
	if (v.family == n) {
		LOG("no graphics queue");
		return false;
	}
	const float prio = 1.0f;
	VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
	qci.queueFamilyIndex = v.family;
	qci.queueCount = 1;
	qci.pQueuePriorities = &prio;
	const char* exts[] = {VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME};
	VkPhysicalDeviceVulkan11Features f11{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES};
	f11.samplerYcbcrConversion = VK_TRUE;
	VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, &f11};
	dci.queueCreateInfoCount = 1;
	dci.pQueueCreateInfos = &qci;
	dci.enabledExtensionCount = 1;
	dci.ppEnabledExtensionNames = exts;
	VKCHECK(vkCreateDevice(v.phys, &dci, nullptr, &v.dev));
#define LOAD_DEVICE(n) n = (PFN_##n)vkGetDeviceProcAddr(v.dev, #n);
	VK_DEVICE_FNS(LOAD_DEVICE)
	vkGetDeviceQueue(v.dev, v.family, 0, &v.queue);
	VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
	pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	pci.queueFamilyIndex = v.family;
	VKCHECK(vkCreateCommandPool(v.dev, &pci, nullptr, &v.pool));
	VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
	cai.commandPool = v.pool;
	cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	cai.commandBufferCount = 1;
	VKCHECK(vkAllocateCommandBuffers(v.dev, &cai, &v.cmd));
	return true;
}

static bool vk_begin(VK& v) {
	VKCHECK(vkResetCommandBuffer(v.cmd, 0));
	VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
	bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
	VKCHECK(vkBeginCommandBuffer(v.cmd, &bi));
	return true;
}

static bool vk_submit_wait(VK& v) {
	VKCHECK(vkEndCommandBuffer(v.cmd));
	VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
	si.commandBufferCount = 1;
	si.pCommandBuffers = &v.cmd;
	VKCHECK(vkQueueSubmit(v.queue, 1, &si, VK_NULL_HANDLE));
	VKCHECK(vkQueueWaitIdle(v.queue));
	return true;
}

struct Buffer {
	VkBuffer buf = VK_NULL_HANDLE;
	VkDeviceMemory mem = VK_NULL_HANDLE;
	void* map = nullptr;
	VkDeviceSize size = 0;
};

static bool make_buffer(VK& v, VkDeviceSize size, bool host, Buffer& b) {
	VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
	bci.size = size;
	bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
	VKCHECK(vkCreateBuffer(v.dev, &bci, nullptr, &b.buf));
	VkMemoryRequirements mr;
	vkGetBufferMemoryRequirements(v.dev, b.buf, &mr);
	const int type = memory_type(v, mr.memoryTypeBits, host ? VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
	                                                         : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (type < 0) {
		LOG("no memory type for a buffer");
		return false;
	}
	VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
	mai.allocationSize = mr.size;
	mai.memoryTypeIndex = (uint32_t)type;
	VKCHECK(vkAllocateMemory(v.dev, &mai, nullptr, &b.mem));
	VKCHECK(vkBindBufferMemory(v.dev, b.buf, b.mem, 0));
	if (host) VKCHECK(vkMapMemory(v.dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
	b.size = size;
	return true;
}

static void free_buffer(VK& v, Buffer& b) {
	if (b.buf) vkDestroyBuffer(v.dev, b.buf, nullptr);
	if (b.mem) vkFreeMemory(v.dev, b.mem, nullptr);
	b = Buffer{};
}

static void barrier(VK& v, VkImage img, bool acquire, bool from_undefined) {
	VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
	b.srcAccessMask = acquire ? 0 : VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
	b.dstAccessMask = acquire ? VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT : 0;
	b.oldLayout = acquire && from_undefined ? VK_IMAGE_LAYOUT_UNDEFINED : VK_IMAGE_LAYOUT_GENERAL;
	b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
	b.srcQueueFamilyIndex = acquire ? VK_QUEUE_FAMILY_EXTERNAL : v.family;
	b.dstQueueFamilyIndex = acquire ? v.family : VK_QUEUE_FAMILY_EXTERNAL;
	b.image = img;
	b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
	vkCmdPipelineBarrier(v.cmd, acquire ? VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT : VK_PIPELINE_STAGE_TRANSFER_BIT,
	                     acquire ? VK_PIPELINE_STAGE_TRANSFER_BIT : VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
}

static std::vector<VkBufferImageCopy> plane_copies(const std::vector<Plane>& planes) {
	std::vector<VkBufferImageCopy> c;
	for (size_t p = 0; p < planes.size(); p++) {
		VkBufferImageCopy r{};
		r.bufferOffset = plane_offset(planes, p);
		r.imageSubresource = {(VkImageAspectFlags)planes[p].aspect, 0, 0, 1};
		r.imageExtent = {(uint32_t)planes[p].vk_width, (uint32_t)planes[p].rows, 1};
		c.push_back(r);
	}
	return c;
}

static bool vk_read(VK& v, VkImage img, const std::vector<Plane>& planes, bool from_undefined, std::vector<uint8_t>& pic) {
	Buffer rb;
	const size_t n = plane_offset(planes, planes.size());
	if (!make_buffer(v, n, true, rb)) return false;
	if (!vk_begin(v)) return false;
	barrier(v, img, true, from_undefined);
	const auto c = plane_copies(planes);
	vkCmdCopyImageToBuffer(v.cmd, img, VK_IMAGE_LAYOUT_GENERAL, rb.buf, (uint32_t)c.size(), c.data());
	barrier(v, img, false, false);
	if (!vk_submit_wait(v)) return false;
	pic.assign((uint8_t*)rb.map, (uint8_t*)rb.map + n);
	free_buffer(v, rb);
	return true;
}

static bool vk_write(VK& v, VkImage img, const std::vector<Plane>& planes, bool from_undefined, const std::vector<uint8_t>& pic) {
	Buffer ub;
	if (!make_buffer(v, pic.size(), true, ub)) return false;
	memcpy(ub.map, pic.data(), pic.size());
	if (!vk_begin(v)) return false;
	barrier(v, img, true, from_undefined);
	const auto c = plane_copies(planes);
	vkCmdCopyBufferToImage(v.cmd, ub.buf, img, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)c.size(), c.data());
	barrier(v, img, false, false);
	if (!vk_submit_wait(v)) return false;
	free_buffer(v, ub);
	return true;
}

static void print_external_support(VK& v, VkFormat fmt, VkExternalMemoryHandleTypeFlagBits type, const char* name) {
	VkPhysicalDeviceExternalImageFormatInfo ext{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_IMAGE_FORMAT_INFO};
	ext.handleType = type;
	VkPhysicalDeviceImageFormatInfo2 fi{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2, &ext};
	fi.format = fmt;
	fi.type = VK_IMAGE_TYPE_2D;
	fi.tiling = VK_IMAGE_TILING_OPTIMAL;
	fi.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	VkExternalImageFormatProperties ep{VK_STRUCTURE_TYPE_EXTERNAL_IMAGE_FORMAT_PROPERTIES};
	VkImageFormatProperties2 p{VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2, &ep};
	const VkResult r = vkGetPhysicalDeviceImageFormatProperties2(v.phys, &fi, &p);
	const VkExternalMemoryFeatureFlags f = ep.externalMemoryProperties.externalMemoryFeatures;
	LOG("  %s: %s, max %ux%u, features%s%s%s", name, r == VK_SUCCESS ? "supported" : "NOT supported", p.imageFormatProperties.maxExtent.width,
	    p.imageFormatProperties.maxExtent.height, f & VK_EXTERNAL_MEMORY_FEATURE_DEDICATED_ONLY_BIT ? " dedicated-only" : "",
	    f & VK_EXTERNAL_MEMORY_FEATURE_EXPORTABLE_BIT ? " exportable" : "", f & VK_EXTERNAL_MEMORY_FEATURE_IMPORTABLE_BIT ? " importable" : "");
}

// ---- One test ------------------------------------------------------------------------------------------------------------------------------
struct Shared {
	Com<ID3D11Texture2D> tex;
	HANDLE handle = nullptr;
	VkImage img = VK_NULL_HANDLE;
	VkDeviceMemory mem = VK_NULL_HANDLE;
};

static bool make_shared(const Options& op, D3D& d, VK& v, int w, int h, Shared& s) {
	const VkFormat fmt = op.r8 ? VK_FORMAT_R8_UNORM : VK_FORMAT_G8_B8R8_2PLANE_420_UNORM;
	const VkExternalMemoryHandleTypeFlagBits type = op.vk_exports ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT : VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT;
	if (!op.vk_exports) {
		D3D11_TEXTURE2D_DESC t = tex_desc(op, w, h);
		t.MiscFlags = D3D11_RESOURCE_MISC_SHARED | D3D11_RESOURCE_MISC_SHARED_NTHANDLE;
		HRCHECK(d.dev->CreateTexture2D(&t, nullptr, s.tex.put()));
		Com<IDXGIResource1> r;
		HRCHECK(s.tex->QueryInterface(__uuidof(IDXGIResource1), (void**)r.put()));
		HRCHECK(r->CreateSharedHandle(nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE, nullptr, &s.handle));
	}
	VkExternalMemoryImageCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO};
	ext.handleTypes = type;
	VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, &ext};
	ici.imageType = VK_IMAGE_TYPE_2D;
	ici.format = fmt;
	ici.extent = {(uint32_t)w, (uint32_t)(op.r8 ? h * 3 / 2 : h), 1};
	ici.mipLevels = 1;
	ici.arrayLayers = 1;
	ici.samples = VK_SAMPLE_COUNT_1_BIT;
	ici.tiling = VK_IMAGE_TILING_OPTIMAL;
	ici.usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
	ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
	VKCHECK(vkCreateImage(v.dev, &ici, nullptr, &s.img));
	VkMemoryDedicatedRequirements ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
	VkMemoryRequirements2 mr{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2, &ded};
	VkImageMemoryRequirementsInfo2 mri{VK_STRUCTURE_TYPE_IMAGE_MEMORY_REQUIREMENTS_INFO_2};
	mri.image = s.img;
	vkGetImageMemoryRequirements2(v.dev, &mri, &mr);
	uint32_t bits = mr.memoryRequirements.memoryTypeBits;
	VkMemoryWin32HandlePropertiesKHR hp{VK_STRUCTURE_TYPE_MEMORY_WIN32_HANDLE_PROPERTIES_KHR};
	if (!op.vk_exports) {
		VKCHECK(vkGetMemoryWin32HandlePropertiesKHR(v.dev, type, s.handle, &hp));
		bits &= hp.memoryTypeBits;
	}
	LOG("  Vulkan image: size %llu, alignment %llu, memory types 0x%x (handle 0x%x), dedicated prefers %u requires %u",
	    (unsigned long long)mr.memoryRequirements.size, (unsigned long long)mr.memoryRequirements.alignment, mr.memoryRequirements.memoryTypeBits,
	    op.vk_exports ? 0 : hp.memoryTypeBits, ded.prefersDedicatedAllocation, ded.requiresDedicatedAllocation);
	int mt = memory_type(v, bits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
	if (mt < 0) mt = memory_type(v, bits, 0);
	if (mt < 0) {
		LOG("no memory type for the shared image");
		return false;
	}
	VkMemoryDedicatedAllocateInfo dai{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
	dai.image = s.img;
	VkImportMemoryWin32HandleInfoKHR imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_WIN32_HANDLE_INFO_KHR, &dai};
	imp.handleType = type;
	imp.handle = s.handle;
	VkExportMemoryWin32HandleInfoKHR exw{VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR, &dai};
	exw.dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
	VkExportMemoryAllocateInfo exa{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO, &exw};
	exa.handleTypes = type;
	VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, op.vk_exports ? (void*)&exa : (void*)&imp};
	mai.allocationSize = mr.memoryRequirements.size;
	mai.memoryTypeIndex = (uint32_t)mt;
	VKCHECK(vkAllocateMemory(v.dev, &mai, nullptr, &s.mem));
	VKCHECK(vkBindImageMemory(v.dev, s.img, s.mem, 0));
	if (op.vk_exports) {
		VkMemoryGetWin32HandleInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
		gi.memory = s.mem;
		gi.handleType = type;
		VKCHECK(vkGetMemoryWin32HandleKHR(v.dev, &gi, &s.handle));
		HRCHECK(d.dev1->OpenSharedResource1(s.handle, __uuidof(ID3D11Texture2D), (void**)s.tex.put()));
		D3D11_TEXTURE2D_DESC t;
		s.tex->GetDesc(&t);
		LOG("  D3D11 opened it as %ux%u format %d bind 0x%x misc 0x%x", t.Width, t.Height, (int)t.Format, t.BindFlags, t.MiscFlags);
	}
	return true;
}

static void free_shared(VK& v, Shared& s) {
	if (s.img) vkDestroyImage(v.dev, s.img, nullptr);
	if (s.mem) vkFreeMemory(v.dev, s.mem, nullptr);
	if (s.handle) CloseHandle(s.handle);
	s.tex.reset();
	s.img = VK_NULL_HANDLE;
	s.mem = VK_NULL_HANDLE;
	s.handle = nullptr;
}

struct Guards {
	std::vector<Buffer> vk;
	std::vector<Com<ID3D11Texture2D>*> d3d;
	~Guards() {
		for (auto* t : d3d) delete t;
	}
};

static uint32_t guard_value(size_t i) { return 0xA55A0000u | (uint32_t)i; }

// n Vulkan device-local buffers and n D3D11 R8 textures, each one picture large, filled with a known value.
static bool make_guards(const Options& op, D3D& d, VK& v, int w, int h, Guards& g) {
	const size_t bytes = (size_t)w * h * 3 / 2;
	for (int i = 0; i < op.guards; i++) {
		Buffer b;
		if (!make_buffer(v, bytes, false, b)) return false;
		if (!vk_begin(v)) return false;
		vkCmdFillBuffer(v.cmd, b.buf, 0, VK_WHOLE_SIZE, guard_value(g.vk.size()));
		if (!vk_submit_wait(v)) return false;
		g.vk.push_back(b);

		D3D11_TEXTURE2D_DESC t = tex_desc(op, w, h);
		t.Format = DXGI_FORMAT_R8_UNORM;
		t.Height = h * 3 / 2;
		std::vector<uint8_t> fill(bytes);
		for (size_t k = 0; k < bytes; k++) fill[k] = (uint8_t)(guard_value(1000 + g.d3d.size()) >> (8 * (k & 3)));
		D3D11_SUBRESOURCE_DATA sd{fill.data(), (UINT)w, 0};
		auto* tex = new Com<ID3D11Texture2D>;
		g.d3d.push_back(tex);
		HRCHECK(d.dev->CreateTexture2D(&t, &sd, tex->put()));
	}
	return true;
}

// Returns the number of damaged guards; prints where the shared picture's rows show up in them.
static int check_guards(const Options& op, D3D& d, VK& v, int w, int h, Guards& g, size_t first, size_t count, const char* when,
                        const std::vector<uint8_t>& pic) {
	int damaged = 0;
	const size_t bytes = (size_t)w * h * 3 / 2;
	for (size_t i = first; i < first + count && i < g.vk.size(); i++) {
		Buffer rb;
		if (!make_buffer(v, bytes, true, rb)) return 1;
		if (!vk_begin(v)) return 1;
		VkBufferCopy c{0, 0, bytes};
		vkCmdCopyBuffer(v.cmd, g.vk[i].buf, rb.buf, 1, &c);
		if (!vk_submit_wait(v)) return 1;
		const uint32_t want = guard_value(i);
		const uint32_t* p = (const uint32_t*)rb.map;
		size_t wrong = 0, first_wrong = 0;
		for (size_t k = 0; k < bytes / 4; k++)
			if (p[k] != want && wrong++ == 0) first_wrong = k * 4;
		if (wrong) {
			damaged++;
			const long long at = find_bytes((const uint8_t*)rb.map, bytes, pic.data(), 32);
			LOG("  guard Vulkan buffer %zu (%s): %zu of %zu words damaged, first at byte %zu; shared picture start %s", i, when, wrong, bytes / 4,
			    first_wrong, at >= 0 ? ("found at byte " + std::to_string(at)).c_str() : "not found");
		}
		free_buffer(v, rb);
	}
	for (size_t i = first; i < first + count && i < g.d3d.size(); i++) {
		std::vector<Plane> one{{w, h * 3 / 2, VK_IMAGE_ASPECT_COLOR_BIT, w}};
		std::vector<uint8_t> got;
		if (!d3d_read(d, *g.d3d[i], one, got)) return 1;
		size_t wrong = 0, first_wrong = 0;
		for (size_t k = 0; k < bytes; k++)
			if (got[k] != (uint8_t)(guard_value(1000 + i) >> (8 * (k & 3))) && wrong++ == 0) first_wrong = k;
		if (wrong) {
			damaged++;
			const long long at = find_bytes(got.data(), bytes, pic.data(), 32);
			LOG("  guard D3D11 texture %zu (%s): %zu of %zu bytes damaged, first at byte %zu; shared picture start %s", i, when, wrong, bytes, first_wrong,
			    at >= 0 ? ("found at byte " + std::to_string(at)).c_str() : "not found");
		}
	}
	(void)op;
	return damaged;
}

static bool run_one(const Options& op, D3D& d, VK& v, int w, int h, int iter, size_t& failures) {
	LOG("--- %dx%d %s, %s exports, iteration %d", w, h, op.r8 ? "R8 (control)" : "NV12", op.vk_exports ? "Vulkan" : "D3D11", iter + 1);
	const auto planes = planes_of(op, w, h);
	Guards g;
	if (!make_guards(op, d, v, w, h, g)) return false; // guards 0..n-1: made before the shared texture
	Shared s;
	if (!make_shared(op, d, v, w, h, s)) {
		free_shared(v, s);
		return false;
	}
	Guards after;
	if (!make_guards(op, d, v, w, h, after)) return false; // made after it

	const uint32_t seed_a = 0x1000u + iter * 2, seed_b = 0x1001u + iter * 2;
	const auto pic_a = make_picture(planes, seed_a), pic_b = make_picture(planes, seed_b);
	std::vector<uint8_t> got;
	size_t bad = 0;

	// 1. D3D11 writes, Vulkan reads.
	if (!d3d_write(d, s.tex, planes, pic_a)) return false;
	{
		std::vector<uint8_t> back;
		if (!d3d_read(d, s.tex, planes, back)) return false;
		bad += compare("D3D11 wrote A, D3D11 reads:", planes, back, pic_a);
	}
	if (!vk_read(v, s.img, planes, op.old_layout_undefined, got)) return false;
	bad += compare("D3D11 wrote A, Vulkan reads:", planes, got, pic_a);

	// 2. Vulkan writes, D3D11 reads.
	if (!vk_write(v, s.img, planes, op.old_layout_undefined, pic_b)) return false;
	if (!d3d_read(d, s.tex, planes, got)) return false;
	bad += compare("Vulkan wrote B, D3D11 reads:", planes, got, pic_b);

	// 3. Vulkan reads its own write.
	if (!vk_read(v, s.img, planes, false, got)) return false;
	bad += compare("Vulkan wrote B, Vulkan reads:", planes, got, pic_b);

	// 4. Guards.
	const int damaged = check_guards(op, d, v, w, h, g, 0, g.vk.size(), "made before", pic_b) +
	                    check_guards(op, d, v, w, h, after, 0, after.vk.size(), "made after", pic_b);
	if (!damaged) LOG("  guards: %d Vulkan buffers and %d D3D11 textures intact", op.guards * 2, op.guards * 2);
	if (bad || damaged) failures++;

	free_shared(v, s);
	for (auto& b : g.vk) free_buffer(v, b);
	for (auto& b : after.vk) free_buffer(v, b);
	return true;
}

int main(int argc, char** argv) {
	Options op;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
		if (a == "--dir") op.vk_exports = next() == "vk";
		else if (a == "--format") op.r8 = next() == "r8";
		else if (a == "--size") {
			int w = 0, h = 0;
			if (sscanf(next().c_str(), "%dx%d", &w, &h) == 2 && w > 0 && h > 0 && w % 2 == 0 && h % 2 == 0) op.sizes.push_back({w, h});
			else {
				LOG("--size wants WxH with even numbers");
				return 2;
			}
		} else if (a == "--iterations") op.iterations = atoi(next().c_str());
		else if (a == "--adapter") op.adapter = atoi(next().c_str());
		else if (a == "--old-layout") op.old_layout_undefined = next() == "undefined";
		else if (a == "--guards") op.guards = atoi(next().c_str());
		else {
			LOG("unknown option %s (see the comment at the top of the source)", a.c_str());
			return 2;
		}
	}
	if (op.sizes.empty()) op.sizes = {{1280, 720}, {1920, 1080}, {3840, 2160}};

	D3D d;
	if (!d3d_init(d, op.adapter)) return 2;
	VK v;
	if (!vk_init(v, d.desc.AdapterLuid)) return 2;
	VkFormatProperties fp;
	vkGetPhysicalDeviceFormatProperties(v.phys, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, &fp);
	LOG("G8_B8R8_2PLANE_420_UNORM optimal tiling features 0x%x (transfer src %d, dst %d, disjoint %d)", fp.optimalTilingFeatures,
	    !!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_SRC_BIT), !!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_TRANSFER_DST_BIT),
	    !!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_DISJOINT_BIT));
	LOG("External memory support (optimal tiling, transfer src/dst + sampled):");
	print_external_support(v, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT, "NV12 D3D11_TEXTURE");
	print_external_support(v, VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT, "NV12 OPAQUE_WIN32");
	print_external_support(v, VK_FORMAT_R8_UNORM, VK_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_TEXTURE_BIT, "R8 D3D11_TEXTURE");
	print_external_support(v, VK_FORMAT_R8_UNORM, VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT, "R8 OPAQUE_WIN32");

	size_t failures = 0, runs = 0;
	for (auto [w, h] : op.sizes)
		for (int it = 0; it < op.iterations; it++, runs++)
			if (!run_one(op, d, v, w, h, it, failures)) {
				LOG("setup failed");
				vkDeviceWaitIdle(v.dev);
				return 2;
			}
	LOG("=== %zu of %zu runs had wrong data or damaged guards", failures, runs);
	vkDeviceWaitIdle(v.dev);
	vkDestroyCommandPool(v.dev, v.pool, nullptr);
	vkDestroyDevice(v.dev, nullptr);
	vkDestroyInstance(v.inst, nullptr);
	return failures ? 1 : 0;
}
