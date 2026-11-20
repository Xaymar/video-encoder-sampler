// Native AMD AMF backend: talks to amfrt64.dll directly (loaded at runtime), no libavcodec.
// Encoder options are raw AMF property names (e.g. QualityPreset, BPicturesPattern for H.264,
// HevcQualityPreset for HEVC) so every property AMF exposes can be swept. Enums are numeric.
#include "encoder.h"

#ifdef _WIN32

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include "formats.h"
#include "options.h"
#include "repack.h"

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>
#include <winternl.h>
#include <d3dkmthk.h>

#include <filesystem>
#include <vector>

#include "gpu.h"
#include "decode.h"
#include "vk_compute.h"
#ifndef VES_NO_VULKAN
#include <core/VulkanAMF.h>
#endif

#include <components/PreProcessing.h>
#include <algorithm>
#include <components/VideoEncoderAV1.h>
#include <components/VideoEncoderHEVC.h>
#include <components/VideoEncoderVCE.h>
#include <core/Factory.h>

using Microsoft::WRL::ComPtr;

namespace {

thread_local bool probing = false; // set while amf_supports() tries a codec out

std::string runtime_used; // "1.4.24 from C:\\...\\amfrt64.dll", once load_runtime() has loaded one in a process that serves one GPU
int dedicated_gpu = -1; // this process serves AMD GPU n only (an encoder process, a probe): its runtime comes from that GPU's driver package

// The folder of the driver package that drives an adapter (System32\DriverStore\FileRepository\<package>), asked of the kernel the way the Vulkan loader finds
// a GPU's own driver; empty when Windows cannot say. The functions are looked up at run time: they are in gdi32, which is not linked otherwise.
std::filesystem::path driver_store_folder(uint64_t luid) {
	HMODULE gdi = LoadLibraryW(L"gdi32.dll");
	auto open_adapter = gdi ? (NTSTATUS(APIENTRY*)(D3DKMT_OPENADAPTERFROMLUID*))GetProcAddress(gdi, "D3DKMTOpenAdapterFromLuid") : nullptr;
	auto query = gdi ? (NTSTATUS(APIENTRY*)(const D3DKMT_QUERYADAPTERINFO*))GetProcAddress(gdi, "D3DKMTQueryAdapterInfo") : nullptr;
	auto close_adapter = gdi ? (NTSTATUS(APIENTRY*)(const D3DKMT_CLOSEADAPTER*))GetProcAddress(gdi, "D3DKMTCloseAdapter") : nullptr;
	if (!open_adapter || !query || !close_adapter) return {};
	D3DKMT_OPENADAPTERFROMLUID open{};
	open.AdapterLuid.LowPart = (DWORD)luid;
	open.AdapterLuid.HighPart = (LONG)(luid >> 32);
	if (open_adapter(&open) != 0) return {};
	std::wstring out;
	D3DDDI_QUERYREGISTRY_INFO base{};
	base.QueryType = D3DDDI_QUERYREGISTRY_DRIVERSTOREPATH;
	D3DKMT_QUERYADAPTERINFO q{};
	q.hAdapter = open.hAdapter;
	q.Type = KMTQAITYPE_QUERYREGISTRY;
	q.pPrivateDriverData = &base;
	q.PrivateDriverDataSize = sizeof base;
	// The first question only tells how long the answer is.
	if (query(&q) == 0 && base.Status == D3DDDI_QUERYREGISTRY_STATUS_BUFFER_OVERFLOW) {
		std::vector<uint8_t> buf(sizeof base + base.OutputValueSize);
		auto* full = (D3DDDI_QUERYREGISTRY_INFO*)buf.data();
		full->QueryType = D3DDDI_QUERYREGISTRY_DRIVERSTOREPATH;
		q.pPrivateDriverData = full;
		q.PrivateDriverDataSize = (UINT)buf.size();
		if (query(&q) == 0 && full->Status == D3DDDI_QUERYREGISTRY_STATUS_SUCCESS) out.assign(full->OutputString, full->OutputValueSize / sizeof(wchar_t));
	}
	D3DKMT_CLOSEADAPTER close{};
	close.hAdapter = open.hAdapter;
	close_adapter(&close);
	while (!out.empty() && out.back() == L'\0') out.pop_back();
	// The kernel may answer with an NT path.
	if (out.rfind(L"\\??\\", 0) == 0) out.erase(0, 4);
	if (_wcsnicmp(out.c_str(), L"\\SystemRoot\\", 12) == 0) {
		wchar_t windows[MAX_PATH];
		const UINT n = GetWindowsDirectoryW(windows, MAX_PATH);
		out = std::wstring(windows, n) + out.substr(11);
	}
	return out;
}

// amfrt64.dll in a driver package: at its top or in one of its folders (AMD keeps it in one); the shallowest wins.
std::filesystem::path find_runtime(const std::filesystem::path& folder) {
	namespace fs = std::filesystem;
	std::error_code ec;
	fs::path best;
	int best_depth = 99;
	for (fs::recursive_directory_iterator it(folder, fs::directory_options::skip_permission_denied, ec), end; !ec && it != end; it.increment(ec)) {
		if (it.depth() > 2) { it.disable_recursion_pending(); continue; }
		if (it->is_regular_file(ec) && _wcsicmp(it->path().filename().c_str(), AMF_DLL_NAME) == 0 && it.depth() < best_depth) { best = it->path(); best_depth = it.depth(); }
	}
	return best;
}

// AMD's runtime for GPU `gpu`. A process that serves that GPU alone takes the runtime of the GPU's own driver package: AMD drops older GPUs from new runtimes
// (Vega is refused with AMF_CODEC_NOT_SUPPORTED by the 2024 ones), while the driver that still runs such a GPU came with a runtime that can encode on it. The newest
// driver installed puts its runtime in System32 for everyone, which is what any other process gets. Loaded once per process.
HMODULE load_runtime(int gpu) {
	static std::mutex m;
	static HMODULE dll = nullptr;
	std::lock_guard lock(m);
	if (dll) return dll;
	std::filesystem::path from;
	if (gpu == dedicated_gpu) {
		const auto gpus = list_gpus(VENDOR_AMD);
		if (gpu >= 0 && gpu < (int)gpus.size()) {
			const auto folder = driver_store_folder(gpus[(size_t)gpu].luid);
			if (!folder.empty()) from = find_runtime(folder);
		}
		if (!from.empty()) {
			SetDllDirectoryW(from.parent_path().c_str()); // what the runtime loads by name comes from the same package
			dll = LoadLibraryExW(from.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
			if (!dll) { fprintf(stderr, "amf: cannot load %s (error %lu); using the system's runtime\n", from.string().c_str(), GetLastError()); SetDllDirectoryW(nullptr); }
		}
	}
	if (!dll) { dll = LoadLibraryW(AMF_DLL_NAME); from = AMF_DLL_NAME; }
	if (dll && gpu == dedicated_gpu) {
		amf_uint64 v = 0;
		auto version = (AMFQueryVersion_Fn)GetProcAddress(dll, AMF_QUERY_VERSION_FUNCTION_NAME);
		if (version) version(&v);
		char b[64];
		snprintf(b, sizeof b, "%u.%u.%u", (unsigned)AMF_GET_MAJOR_VERSION(v), (unsigned)AMF_GET_MINOR_VERSION(v), (unsigned)AMF_GET_SUBMINOR_VERSION(v));
		runtime_used = std::string(b) + " from " + from.string();
		fprintf(stderr, "amf: GPU %d uses runtime %s\n", gpu, runtime_used.c_str());
	}
	return dll;
}

struct Names { // property names differ per codec
	const wchar_t *component, *usage, *target, *peak, *size, *rate, *rc, *gop, *bitdepth;
	int64_t usage_transcoding, rc_cbr;
};

const Names H264 = {AMFVideoEncoderVCE_AVC, AMF_VIDEO_ENCODER_USAGE, AMF_VIDEO_ENCODER_TARGET_BITRATE, AMF_VIDEO_ENCODER_PEAK_BITRATE,
	AMF_VIDEO_ENCODER_FRAMESIZE, AMF_VIDEO_ENCODER_FRAMERATE, AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD, AMF_VIDEO_ENCODER_IDR_PERIOD, AMF_VIDEO_ENCODER_COLOR_BIT_DEPTH,
	AMF_VIDEO_ENCODER_USAGE_TRANSCODING, AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR};
const Names HEVC = {AMFVideoEncoder_HEVC, AMF_VIDEO_ENCODER_HEVC_USAGE, AMF_VIDEO_ENCODER_HEVC_TARGET_BITRATE, AMF_VIDEO_ENCODER_HEVC_PEAK_BITRATE,
	AMF_VIDEO_ENCODER_HEVC_FRAMESIZE, AMF_VIDEO_ENCODER_HEVC_FRAMERATE, AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD, AMF_VIDEO_ENCODER_HEVC_GOP_SIZE, AMF_VIDEO_ENCODER_HEVC_COLOR_BIT_DEPTH,
	AMF_VIDEO_ENCODER_HEVC_USAGE_TRANSCODING, AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CBR};
const Names AV1 = {AMFVideoEncoder_AV1, AMF_VIDEO_ENCODER_AV1_USAGE, AMF_VIDEO_ENCODER_AV1_TARGET_BITRATE, AMF_VIDEO_ENCODER_AV1_PEAK_BITRATE,
	AMF_VIDEO_ENCODER_AV1_FRAMESIZE, AMF_VIDEO_ENCODER_AV1_FRAMERATE, AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD, AMF_VIDEO_ENCODER_AV1_GOP_SIZE, AMF_VIDEO_ENCODER_AV1_COLOR_BIT_DEPTH,
	AMF_VIDEO_ENCODER_AV1_USAGE_TRANSCODING, AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CBR};

std::wstring wide(const std::string& s) { return std::wstring(s.begin(), s.end()); } // property names are ASCII

// AMF's trace (warnings and errors, and its debug lines) into the log while one is alive: InitVulkan only returns AMF_FAIL and says why there.
class TraceToLog : public amf::AMFTraceWriter {
public:
	explicit TraceToLog(amf::AMFFactory* f) {
		if (!f || f->GetTrace(&trace) != AMF_OK || !trace) { trace = nullptr; return; }
		trace->RegisterWriter(kId, this, true);
		trace->TraceEnableAsync(false); // lines arrive before a crash
		trace->SetWriterLevel(kId, AMF_TRACE_DEBUG);
		level = trace->SetGlobalLevel(AMF_TRACE_DEBUG);
	}
	~TraceToLog() {
		if (!trace) return;
		trace->TraceFlush();
		trace->SetGlobalLevel(level);
		trace->UnregisterWriter(kId);
	}
	void AMF_CDECL_CALL Write(const wchar_t* scope, const wchar_t* message) override {
		std::string line = "amf trace: ";
		for (const wchar_t* p = scope; p && *p; p++) line += *p < 128 ? (char)*p : '?';
		line += ": ";
		for (const wchar_t* p = message; p && *p; p++) if (*p != L'\r' && *p != L'\n') line += *p < 128 ? (char)*p : '?';
		fprintf(stderr, "%s\n", line.c_str());
		fflush(stderr);
	}
	void AMF_CDECL_CALL Flush() override { fflush(stderr); }
private:
	static constexpr const wchar_t* kId = L"ves";
	amf::AMFTrace* trace = nullptr;
	amf_int32 level = AMF_TRACE_INFO;
};

class Amf : public Encoder {
	HMODULE dll = nullptr;
	amf::AMFFactory* factory = nullptr;
	ComPtr<ID3D11Device> device;
	amf::AMFContextPtr ctx;
	amf::AMFComponentPtr enc;
	int depth = 8;
	Sink sink;
	int w = 0, h = 0;   // the picture
	int cw = 0, ch = 0; // what is coded: the picture and the padding at the right and bottom (AV1: 64x16 blocks)
	// Finished packets are collected on a thread of their own; AMF has no blocking read, so it polls.
	std::thread rx;
	std::atomic<bool> eof{false}, stop{false};
	int b_delay = 0;
	uint64_t adapter_luid = 0; // of the D3D11 adapter the encoder runs on
#ifndef VES_NO_VULKAN
	amf::AMFVulkanDevice vkdev{}; // what InitVulkan() was given (see init_vulkan_on)
#endif
#ifndef VES_NO_VULKAN
	// Frames straight from VRAM (VES_ENC_GPU_INPUT, on by default): the converter's output (a buffer frame on FFmpeg's Vulkan device) is packed into
	// NV12 / P010 by a compute shader (enc_pack, as repack() does in memory, padding included) and copied into a Vulkan surface that AMF allocated on
	// that same device (the context is initialized for Vulkan as well as D3D11), which AMF then converts to D3D11 itself. The encoder is the same
	// D3D11 one as for frames from memory. On AMD this sharing is fine; it is NVIDIA's drivers that corrupt VRAM with shared NV12 images.
	int gstate = 0; // 0: not tried yet, 1: frames from VRAM, -1: through memory
	int gdevice = -1;
	std::unique_ptr<Vk> gvk;
	Buf gstage, gpacked;
	VkDeviceSize gstage_size = 0;
	Pipe gpipe;
	bool gpipe_made = false;
	uint32_t gpitch = 0;
	VkDeviceSize guv_off = 0;

	// InitVulkan on FFmpeg's device (nat), on `c` as it is: "" when it worked, else why not. The compute queue AMF submits to is the last one of the family,
	// which the converter and the scorer (the first) do not use.
	std::string init_vulkan_on(amf::AMFContextPtr& c, const VkNative& nat) {
		amf::AMFContext1Ptr c1(c);
		if (!c1) return "the AMF runtime has no Vulkan support (AMFContext1)";
		amf_size n = 0;
		if (c1->GetVulkanDeviceExtensions(&n, nullptr) == AMF_OK && n) {
			std::vector<const char*> need(n);
			std::string missing;
			if (c1->GetVulkanDeviceExtensions(&n, need.data()) == AMF_OK)
				for (amf_size i = 0; i < n; i++)
					if (need[i] && std::find(nat.extensions.begin(), nat.extensions.end(), std::string(need[i])) == nat.extensions.end())
						missing += (missing.empty() ? "" : ", ") + std::string(need[i]);
			if (!missing.empty()) return "FFmpeg's Vulkan device lacks " + missing + ", which AMF needs";
		}
		c->SetProperty(AMF_CONTEXT_VULKAN_COMPUTE_QUEUE, (amf_int64)(nat.compute_queues - 1));
		// AMF may keep the pointer, not a copy of the struct (a stale one would give it a garbage VkDevice: "vkCreateImage: Invalid device"): it lives as long as the encoder
		vkdev = amf::AMFVulkanDevice{};
		vkdev.cbSizeof = sizeof vkdev;
		vkdev.hInstance = nat.inst;
		vkdev.hPhysicalDevice = nat.phys;
		vkdev.hDevice = nat.dev;
		AMF_RESULT ir;
		{
			TraceToLog trace(factory); // AMF says why in its trace only
			ir = c1->InitVulkan(&vkdev);
		}
		if (ir != AMF_OK) return "InitVulkan failed (result " + std::to_string((int)ir) + ")";
		return "";
	}

	bool gpu_open(const VkDecodedFrame& bf) {
		gstate = -1;
		auto no = [&](const std::string& why) {
			fprintf(stderr, "amf: frames come through memory: %s\n", why.c_str());
			gvk.reset();
			return false;
		};
		if (const char* e = getenv("VES_ENC_GPU_INPUT"); e && atoi(e) == 0) return no("VES_ENC_GPU_INPUT=0");
		std::string why;
		const auto devs = Vk::devices(why);
		if (bf.device < 0 || bf.device >= (int)devs.size()) return no("the frames' Vulkan device is unknown");
		const VkDeviceInfo& di = devs[(size_t)bf.device];
		if (!di.luid_valid || di.luid != adapter_luid) return no("the frames are on another GPU (" + di.name + ")");
		VkNative nat;
		if (!vk_native(bf.device, nat)) return no("the frames' Vulkan device is not FFmpeg's");
		// AMF submits to a compute queue of its own: the last one of the family, which the converter and the scorer (the first) do not use
		if (nat.compute_queues < 2) return no("FFmpeg's Vulkan device has only one compute queue");
		if (const std::string e = init_vulkan_on(ctx, nat); !e.empty()) {
			// Which part AMF does not take: FFmpeg's device on a context that already has D3D11, FFmpeg's device on a context of its own, or a device AMF makes itself.
			auto alone = [&](bool own_device) {
				amf::AMFContextPtr c;
				if (factory->CreateContext(&c) != AMF_OK) return std::string("no context");
				amf::AMFContext1Ptr x(c);
				if (!x) return std::string("no AMFContext1");
				TraceToLog trace(factory);
				amf::AMFVulkanDevice vd{};
				vd.cbSizeof = sizeof vd;
				vd.hInstance = nat.inst;
				vd.hPhysicalDevice = nat.phys;
				vd.hDevice = nat.dev;
				const AMF_RESULT r = x->InitVulkan(own_device ? nullptr : &vd);
				x->Terminate();
				return std::to_string((int)r);
			};
			return no(e + " (on a context without D3D11: " + alone(false) + "; with a device AMF makes: " + alone(true) + ")");
		}
		try {
			gvk = std::make_unique<Vk>(bf.device, 1);
			const int bps = depth > 8 ? 2 : 1;
			gpitch = (uint32_t)(((size_t)cw * bps + 255) & ~(size_t)255);
			guv_off = (VkDeviceSize)gpitch * ch;
			gpacked = gvk->buffer(guv_off + (VkDeviceSize)gpitch * ((ch + 1) / 2), 1);
		} catch (const std::exception& e) { return no(e.what()); }
		gdevice = bf.device;
		gstate = 1;
		fprintf(stderr, "amf: frames come straight from VRAM (Vulkan device %d, packed on the GPU into a Vulkan surface of AMF's)\n", bf.device);
		return true;
	}

	// A buffer frame into a Vulkan surface of AMF's, ready when this returns. Null (the run fails) when that does not work.
	amf::AMFSurfacePtr gpu_surface(const VkDecodedFrame& bf) {
		const int bytes = depth > 8 ? 2 : 1;
		const int pw = (w + 1) / 2, ph = (h + 1) / 2;
		if (bf.device != gdevice || bf.nplanes != 3 || bf.bytes != bytes || (int)bf.planes[0].w != w || (int)bf.planes[0].h != h || (int)bf.planes[1].w != pw ||
		    (int)bf.planes[1].h != ph || (int)bf.planes[2].w != pw || (int)bf.planes[2].h != ph) {
			fprintf(stderr, "amf: a frame from VRAM is not the %dx%d %d-bit 4:2:0 the encoder takes\n", w, h, depth);
			return nullptr;
		}
		amf::AMFSurfacePtr surf;
		AMF_RESULT r = ctx->AllocSurface(amf::AMF_MEMORY_VULKAN, depth > 8 ? amf::AMF_SURFACE_P010 : amf::AMF_SURFACE_NV12, cw, ch, &surf);
		if (r != AMF_OK || !surf) { fail(r, "AllocSurface (Vulkan)"); return nullptr; }
		// a plane's native object is a view of the surface's one image (both planes of NV12)
		amf::AMFVulkanView* view = (amf::AMFVulkanView*)surf->GetPlaneAt(0)->GetNative();
		amf::AMFVulkanSurface* vs = view ? view->pSurface : nullptr;
		if (!vs || !vs->hImage) { fprintf(stderr, "amf: the Vulkan surface has no image\n"); return nullptr; }
		VkDeviceSize need = 0;
		for (int i = 0; i < 3; i++) need = std::max<VkDeviceSize>(need, bf.buf_offset[i] + (VkDeviceSize)bf.planes[i].w * bf.planes[i].h * bytes);
		need = (need + 3) & ~(VkDeviceSize)3;
		try {
			if (gstage_size < need) {
				if (gpipe_made) { gvk->release(gpipe); gpipe_made = false; }
				gstage = gvk->buffer(need, 1);
				gstage_size = need;
			}
			if (!gpipe_made) { gpipe = gvk->pipeline("enc_pack", {&gstage, &gpacked}); gpipe_made = true; }
			struct { int32_t w, h, bytes, shift, mode, src_a, src_b, dst_base, pitch_words, ow, oh; } pc{};
			pc.bytes = bytes;
			pc.shift = depth > 8 ? 16 - depth : 0;
			pc.pitch_words = (int32_t)(gpitch / 4);
			auto run = [&](int mode, int pw_, int ph_, int ow, int oh, VkDeviceSize a, VkDeviceSize b, VkDeviceSize dst) {
				pc.w = pw_; pc.h = ph_; pc.ow = ow; pc.oh = oh; pc.mode = mode;
				pc.src_a = (int32_t)a; pc.src_b = (int32_t)b; pc.dst_base = (int32_t)(dst / 4);
				const int row = mode == 1 ? ow * 2 : ow, per = 4 / bytes;
				gvk->bind(gpipe);
				gvk->push(&pc, sizeof pc);
				gvk->dispatch((uint32_t)(((row + per - 1) / per + 15) / 16), (uint32_t)((oh + 15) / 16));
			};
			gvk->begin(0);
			gvk->wait_timeline(0, bf.buf_sem, bf.buf_value);
			if (vs->Sync.hSemaphore && vs->Sync.bSubmitted) { // AMF's last use of the image (a binary semaphore: the value is ignored)
				gvk->wait_timeline(0, vs->Sync.hSemaphore, 0);
				vs->Sync.bSubmitted = false;
			}
			gvk->copy_rows(bf.buffer, 0, need, gstage.buf, 0, need, 1);
			gvk->barrier();
			run(0, w, h, cw, ch, bf.buf_offset[0], 0, 0);
			run(1, pw, ph, cw / 2, ch / 2, bf.buf_offset[1], bf.buf_offset[2], guv_off);
			gvk->barrier();
			VkImageLayout layout = (VkImageLayout)vs->eCurrentLayout;
			uint32_t access = 0;
			gvk->image_to_transfer_dst(vs->hImage, layout, access);
			gvk->copy_to_image(gpacked, 0, gpitch / bytes, VkImagePlane{vs->hImage, VK_IMAGE_ASPECT_PLANE_0_BIT, (uint32_t)cw, (uint32_t)ch});
			gvk->copy_to_image(gpacked, guv_off, gpitch / (2 * bytes), VkImagePlane{vs->hImage, VK_IMAGE_ASPECT_PLANE_1_BIT, (uint32_t)(cw / 2), (uint32_t)(ch / 2)});
			gvk->submit(0);
			gvk->wait(0); // done: AMF need not wait for anything (bSubmitted stays false)
			vs->eCurrentLayout = (amf_uint32)layout;
		} catch (const std::exception& e) {
			fprintf(stderr, "amf: packing a frame on the GPU: %s\n", e.what());
			return nullptr;
		}
		return surf;
	}
#endif

	bool fail(AMF_RESULT r, const char* what) {
		if (probing) return false; // asking whether a codec works is not an error when it does not
		fprintf(stderr, "amf: %s failed (result %d)\n", what, (int)r);
		return false;
	}

	// Set a property from text, converting by the type AMF reports for it.
	bool set(const std::string& key, const std::string& val) {
		std::wstring k = wide(key);
		const amf::AMFPropertyInfo* info = nullptr;
		if (enc->GetPropertyInfo(k.c_str(), &info) != AMF_OK || !info) { fprintf(stderr, "amf: unknown property '%s'\n", key.c_str()); return false; }
		amf::AMFVariant v;
		switch (info->type) {
		case amf::AMF_VARIANT_BOOL: v = (val == "1" || val == "true"); break;
		case amf::AMF_VARIANT_INT64: { char* end = nullptr; const long long n = strtoll(val.c_str(), &end, 10); if (end == val.c_str() || *end) { fprintf(stderr, "amf: '%s' is not a number or a known name for '%s'\n", val.c_str(), key.c_str()); return false; } v = (amf_int64)n; break; }
		case amf::AMF_VARIANT_DOUBLE: { char* end = nullptr; const double d = strtod(val.c_str(), &end); if (end == val.c_str() || *end) { fprintf(stderr, "amf: '%s' is not a number for '%s'\n", val.c_str(), key.c_str()); return false; } v = d; break; }
		case amf::AMF_VARIANT_RATE: { unsigned n = 0, d = 1; sscanf(val.c_str(), "%u/%u", &n, &d); v = AMFConstructRate(n, d); break; }
		case amf::AMF_VARIANT_SIZE: { int a = 0, b = 0; sscanf(val.c_str(), "%dx%d", &a, &b); v = AMFConstructSize(a, b); break; }
		default: fprintf(stderr, "amf: property '%s' has an unsupported type\n", key.c_str()); return false;
		}
		AMF_RESULT r = enc->SetProperty(k.c_str(), v);
		if (r != AMF_OK) { fprintf(stderr, "amf: cannot set '%s' to '%s' (result %d)\n", key.c_str(), val.c_str(), (int)r); return false; }
		return true;
	}

	// Planar frame into NV12 / P010 (interleaved chroma); 10-bit samples go in the top bits of 16, so they are shifted left by 6.
	void repack(const Picture* fr, amf::AMFSurface* surf) const {
		amf::AMFPlane *yp = surf->GetPlaneAt(0), *uvp = surf->GetPlaneAt(1);
		pack_plane(*fr, 0, w, h, depth, (uint8_t*)yp->GetNative(), yp->GetHPitch());
		pack_uv(*fr, w, h, depth, (uint8_t*)uvp->GetNative(), uvp->GetHPitch());
		if (cw == w && ch == h) return;
		const int bytes = depth > 8 ? 2 : 1;
		pad_edges((uint8_t*)yp->GetNative(), yp->GetHPitch(), w, h, cw, ch, bytes);
		pad_edges((uint8_t*)uvp->GetNative(), uvp->GetHPitch(), (w + 1) / 2, (h + 1) / 2, cw / 2, ch / 2, 2 * bytes);
	}

	// What the encoder says it can do: where it runs (dedicated hardware, GPU compute or software) and the capability values the SDK lists (maximum bit rate, B-frames,
	// hardware instances, ...), as a JSON object. Names and values are the SDK's own.
	std::string caps_json = "null";
	void read_caps() {
		amf::AMFCapsPtr caps;
		if (enc->GetCaps(&caps) != AMF_OK || !caps) return;
		const amf::AMF_ACCELERATION_TYPE accel = caps->GetAccelerationType();
		caps_json = std::string("{\"ACCELERATION_TYPE\":\"") + (accel == amf::AMF_ACCEL_HARDWARE ? "HARDWARE" : accel == amf::AMF_ACCEL_GPU ? "GPU" : accel == amf::AMF_ACCEL_SOFTWARE ? "SOFTWARE" : "NOT_SUPPORTED") + "\"";
		for (amf_size i = 0; i < caps->GetPropertyCount(); i++) {
			wchar_t name[128];
			amf::AMFVariant v;
			if (caps->GetPropertyAt(i, name, 128, &v) != AMF_OK) continue;
			std::string key, val;
			for (const wchar_t* c = name; *c; c++) key += (char)*c;
			if (key.find("Requested") != std::string::npos) continue; // what is running on the encoder right now: changes from one ask to the next, a capability does not
			if (v.type == amf::AMF_VARIANT_BOOL) val = v.ToBool() ? "true" : "false";
			else if (v.type == amf::AMF_VARIANT_INT64) val = std::to_string(v.ToInt64());
			else continue;
			caps_json += ",\"" + key + "\":" + val;
		}
		caps_json += "}";
	}

	// Property names are written snake_case without the codec's prefix (QualityPreset, HevcQualityPreset and Av1QualityPreset are all quality_preset).
	void collect_names(std::map<std::string, std::string>& names) const {
		for (amf_size i = 0; i < enc->GetPropertiesInfoCount(); i++) {
			const amf::AMFPropertyInfo* pi = nullptr;
			if (enc->GetPropertyInfo(i, &pi) != AMF_OK || !pi || !pi->name) continue;
			std::string native, bare;
			for (const wchar_t* w = pi->name; *w; w++) native += (char)*w;
			bare = native;
			if (native.find("HalfPixel") != std::string::npos || native.find("QuarterPixel") != std::string::npos) continue; // GCN4 and older only, which AMD no longer supports
			// (AdaptiveMiniGOP is current: AMF_Video_Encode_API.md lists it, adaptive B-frames that need pre-analysis with a lookahead of at least 1)
			// Rate control settings are scoped per temporal layer and quality layer ("TL0.QL0.EnforceHRD"); ves uses the base layer only.
			if (native.compare(0, 8, "TL0.QL0.") == 0) bare = native.substr(8);
			else if (native.compare(0, 2, "TL") == 0 && native.find(".QL") != std::string::npos) continue;
			for (const char* prefix : {"Hevc", "Av1", "AV1"}) {
				const size_t n = strlen(prefix);
				if (bare.size() > n && bare.compare(0, n, prefix) == 0 && isupper((unsigned char)bare[n])) { bare.erase(0, n); break; }
			}
			std::string name = snake_case(bare);
			if (name == "quality_preset") name = "preset"; // AMF's QualityPreset, HevcQualityPreset and Av1QualityPreset
			if (name == "profile_level") name = "level";  // ProfileLevel and HevcProfileLevel; the AV1 one is Av1Level already
			if (name == "rate_control_preanalysis_enable" || name == "rate_control_pre_analysis_enable" || name == "rate_control_pre_encode") name = "pre_encode"; // pre-encode assisted rate control (AV1: Av1RateControlPreEncode)
			names.emplace(name, native);
		}
	}

	// The property names (see collect_names) of the last open(), before pre-analysis was switched on and after.
	std::map<std::string, std::string> names_before_pa, names_all;

	static std::string narrow(const wchar_t* w) {
		std::string out;
		for (; w && *w; w++) out += *w < 128 ? (char)*w : '?';
		return out;
	}
	static std::string text(const amf::AMFVariantStruct& v) {
		char b[64];
		switch (v.type) {
		case amf::AMF_VARIANT_BOOL: return v.boolValue ? "true" : "false";
		case amf::AMF_VARIANT_INT64: return std::to_string(v.int64Value);
		case amf::AMF_VARIANT_DOUBLE: snprintf(b, sizeof b, "%g", v.doubleValue); return b;
		case amf::AMF_VARIANT_FLOAT: snprintf(b, sizeof b, "%g", (double)v.floatValue); return b;
		case amf::AMF_VARIANT_SIZE: return std::to_string(v.sizeValue.width) + "x" + std::to_string(v.sizeValue.height);
		case amf::AMF_VARIANT_RATE: return std::to_string(v.rateValue.num) + "/" + std::to_string(v.rateValue.den);
		default: return {};
		}
	}

public:
	// The value of every property after open(), by option name.
	std::vector<std::pair<std::string, std::string>> settings() override {
		std::vector<std::pair<std::string, std::string>> out;
		if (!enc) return out;
		for (auto& [name, native] : names_all) {
			amf::AMFVariant v;
			if (enc->GetProperty(wide(native).c_str(), &v) != AMF_OK) continue;
			std::string t = v.type == amf::AMF_VARIANT_BOOL ? (v.ToBool() ? "1" : "0") : text(v);
			if (!t.empty()) out.push_back({name, t});
		}
		return out;
	}

	// For `ves --list-options`: every property the encoder of the last open() offers that can be written, with the range, names and default it gives for it.
	OptionList options() const {
		OptionList out;
		for (auto& [name, native] : names_all) {
			const amf::AMFPropertyInfo* pi = nullptr;
			if (enc->GetPropertyInfo(wide(native).c_str(), &pi) != AMF_OK || !pi) continue;
			if (!(pi->accessType & amf::AMF_PROPERTY_ACCESS_WRITE)) continue;
			std::string values;
			if (pi->type == amf::AMF_VARIANT_BOOL) values = "true | false";
			else if (pi->pEnumDescription) {
				for (const amf::AMFEnumDescriptionEntry* e = pi->pEnumDescription; e->name; e++) values += (values.empty() ? "" : " | ") + std::to_string(e->value) + " " + narrow(e->name);
			} else if (pi->type == amf::AMF_VARIANT_INT64 || pi->type == amf::AMF_VARIANT_DOUBLE) {
				const std::string lo = text(pi->minValue), hi = text(pi->maxValue);
				values = !lo.empty() && !hi.empty() && lo != hi ? lo + ".." + hi : "number";
			} else if (pi->type == amf::AMF_VARIANT_SIZE) values = "WxH";
			else if (pi->type == amf::AMF_VARIANT_RATE) values = "num/den";
			else continue; // set() cannot write it
			std::string about = narrow(pi->desc);
			if (about.empty()) about = native;
			else about = native + ": " + about;
			if (const std::string d = text(pi->defaultValue); !d.empty()) about += " (default " + d + ")";
			if (!names_before_pa.count(name)) about += "; only with enable_pre_analysis on";
			out.push_back({name, values, about});
		}
		out.push_back({"pre_filter", "0 | 1", "AMF's adaptive pre-encode filter (a denoiser tuned by the bitrate), a component in front of the encoder"});
		out.push_back({"pre_filter_strength", "0..10", "default 4"});
		out.push_back({"pre_filter_sensitivity", "0..10", "lower keeps more detail; default 4"});
		return out;
	}

private:
	// The pre-encode filter: AMF's adaptive filter (a denoiser tuned by the target bitrate), run on every picture before the encoder gets it.
	amf::AMFComponentPtr filter_pp;
	bool open_filter(const std::map<std::string, std::string>& opts, const Params& p) {
		bool on = false;
		amf_int64 strength = 4, sensitivity = 4;
		for (auto& [k, v] : opts) {
			char* end = nullptr;
			const long long n = v == "true" ? 1 : v == "false" ? 0 : strtoll(v.c_str(), &end, 10);
			if (v != "true" && v != "false" && (end == v.c_str() || *end)) { fprintf(stderr, "amf: '%s' is not a number for '%s'\n", v.c_str(), k.c_str()); return false; }
			if (k == "pre_filter") on = n != 0;
			else if (k == "pre_filter_strength") strength = n;       // 0 to 10
			else if (k == "pre_filter_sensitivity") sensitivity = n; // 0 to 10, lower keeps more detail
			else { fprintf(stderr, "amf: unknown option '%s'; the pre-encode filter has: pre_filter, pre_filter_strength, pre_filter_sensitivity\n", k.c_str()); return false; }
		}
		if (!on) return true;
		AMF_RESULT r = factory->CreateComponent(ctx, AMFPreProcessing, &filter_pp);
		if (r != AMF_OK) return fail(r, "CreateComponent(PreProcessing)");
		// --check found the strength and sensitivity changing nothing, so a property the component refuses is reported.
		const auto refused = [](AMF_RESULT sr, const char* name) { if (sr != AMF_OK) fprintf(stderr, "amf: the pre-encode filter refused %s (result %d)\n", name, (int)sr); };
		refused(filter_pp->SetProperty(AMF_PP_ADAPTIVE_FILTER_ENABLE, true), "PPAdaptiveFilterEnable");
		refused(filter_pp->SetProperty(AMF_PP_ADAPTIVE_FILTER_STRENGTH, strength), "PPAdaptiveFilterStrength");
		refused(filter_pp->SetProperty(AMF_PP_ADAPTIVE_FILTER_SENSITIVITY, sensitivity), "PPAdaptiveFilterSensitivity");
		refused(filter_pp->SetProperty(AMF_PP_TARGET_BITRATE, (amf_int64)p.bitrate), "PPTargetBitrate");
		refused(filter_pp->SetProperty(AMF_PP_FRAME_RATE, AMFConstructRate(p.framerate.num, p.framerate.den)), "PPFrameRate");
		if ((r = filter_pp->Init(depth > 8 ? amf::AMF_SURFACE_P010 : amf::AMF_SURFACE_NV12, cw, ch)) != AMF_OK) return fail(r, "Init(PreProcessing)");
		return true;
	}

	// The picture the encoder gets: the filtered one when the filter is on.
	bool filter(amf::AMFSurface* in, amf::AMFDataPtr& out) {
		AMF_RESULT r = AMF_OK;
		for (int i = 0; i < 5000; i++) { // AMF_INPUT_FULL: the filter has not taken the last picture's output yet
			r = filter_pp->SubmitInput(in);
			if (r != AMF_INPUT_FULL) break;
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		if (r != AMF_OK) return fail(r, "SubmitInput(PreProcessing)");
		for (int i = 0; i < 5000; i++) {
			r = filter_pp->QueryOutput(&out);
			if (r == AMF_OK && out) { out->SetPts(in->GetPts()); return true; }
			if (r != AMF_OK && r != AMF_REPEAT) return fail(r, "QueryOutput(PreProcessing)");
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		return fail(AMF_REPEAT, "QueryOutput(PreProcessing) timed out");
	}

	void receive() {
		while (!stop && !eof) {
			amf::AMFDataPtr data;
			AMF_RESULT r = enc->QueryOutput(&data);
			if (r == AMF_EOF) { eof = true; return; }
			if (r != AMF_OK || !data) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; } // AMF_REPEAT: nothing ready yet
			amf::AMFBufferPtr buf(data);
			stats.packets++; stats.bytes += buf->GetSize();
			if (sink) sink((const uint8_t*)buf->GetNative(), buf->GetSize(), data->GetPts(), false, NO_DTS);
		}
	}

public:
	const std::string& caps() const { return caps_json; } // see read_caps
	~Amf() override {
		stop = true;
		if (rx.joinable()) rx.join();
		if (filter_pp) filter_pp->Terminate();
		filter_pp = nullptr;
		if (enc) enc->Terminate();
		enc = nullptr;
		if (ctx) ctx->Terminate();
		ctx = nullptr;
		device.Reset();
		// shortcut: amfrt64.dll stays loaded; unloading it while AMF's worker threads wind down has been unreliable
	}


	// The colour of the pictures goes to the encoder as input and as output description, so that the stream says what the pictures are. AMF's
	// defaults are studio range and "unknown", which makes a full range source come out wrong. The properties are named differently per codec.
	void set_color(const Color& c) {
		auto first = [&](std::initializer_list<const wchar_t*> names, const amf::AMFVariant& v) {
			for (const wchar_t* name : names) {
				const amf::AMFPropertyInfo* info = nullptr;
				if (enc->GetPropertyInfo(name, &info) != AMF_OK || !info) continue;
				if (info->type == amf::AMF_VARIANT_BOOL) { amf::AMFVariant b((bool)v.ToInt64()); enc->SetProperty(name, b); }
				else enc->SetProperty(name, v);
				return;
			}
		};
		const bool full = color_full(c);
		first({L"InputFullRangeColor", L"HevcInputFullRangeColor", L"Av1InputFullRangeColor"}, amf::AMFVariant((amf_int64)full));
		first({L"FullRangeColor", L"HevcNominalRange", L"Av1NominalRange"}, amf::AMFVariant((amf_int64)full));
		// Without a description in the source AMF would write its own (BT.709); the stream should say nothing instead, as the others do.
		const amf_int64 primaries = c.primaries ? c.primaries : 2, trc = c.trc ? c.trc : 2, matrix = c.space ? c.space : 2;
		// the converter profile names matrix and range together
		amf_int64 profile = -1;
		if (c.space == 1) profile = full ? 7 : 1;                      // BT.709
		else if (c.space == 5 || c.space == 6) profile = full ? 3 : 0; // BT.601
		else if (c.space == 9 || c.space == 10) profile = full ? 8 : 2; // BT.2020
		if (profile >= 0) {
			first({L"InColorProfile", L"HevcInColorProfile", L"Av1InputColorProfile"}, amf::AMFVariant(profile));
			first({L"OutColorProfile", L"HevcOutColorProfile", L"Av1OutputColorProfile"}, amf::AMFVariant(profile));
		}
		first({L"InColorPrimaries", L"HevcInColorPrimaries", L"Av1InputColorPrimaries"}, amf::AMFVariant(primaries));
		first({L"OutColorPrimaries", L"HevcOutColorPrimaries", L"Av1OutputColorPrimaries"}, amf::AMFVariant(primaries));
		first({L"InColorTransferChar", L"HevcInColorTransferChar", L"Av1InputColorTransferChar"}, amf::AMFVariant(trc));
		first({L"OutColorTransferChar", L"HevcOutColorTransferChar", L"Av1OutputColorTransferChar"}, amf::AMFVariant(trc));
		first({L"InMatrixCoeff", L"HevcInMatrixCoeff", L"Av1InMatrixCoeff"}, amf::AMFVariant(matrix));
		first({L"OutMatrixCoeff", L"HevcOutMatrixCoeff", L"Av1OutMatrixCoeff"}, amf::AMFVariant(matrix));
	}

	bool open(const Params& p, Sink s) override {
		sink = std::move(s);
		w = cw = p.width; h = ch = p.height;
		const Fmt fmt = p.pix;
		depth = fmt.depth;
		if (fmt.chroma != 420 || depth > 10) { fprintf(stderr, "amf: cannot encode %d-bit %d\n", depth, fmt.chroma); return false; }
		const Names* n = p.codec == "h264" ? &H264 : p.codec == "hevc" ? &HEVC : p.codec == "av1" ? &AV1 : nullptr;
		if (!n) { fprintf(stderr, "amf: unknown codec '%s'\n", p.codec.c_str()); return false; }
		Options o = p.options;
		int gpu = 0;
		if (auto it = o.find("gpu"); it != o.end()) { gpu = atoi(it->second.c_str()); o.erase(it); }

		dll = load_runtime(gpu);
		auto init = dll ? (AMFInit_Fn)GetProcAddress(dll, AMF_INIT_FUNCTION_NAME) : nullptr;
		if (!init) { fprintf(stderr, "amf: amfrt64.dll not found\n"); return false; }
		AMF_RESULT r = init(AMF_FULL_VERSION, &factory);
		if (r != AMF_OK) { // an older runtime (a GPU's own, see load_runtime) may want to be asked for its own version
			amf_uint64 v = 0;
			auto version = (AMFQueryVersion_Fn)GetProcAddress(dll, AMF_QUERY_VERSION_FUNCTION_NAME);
			if (version && version(&v) == AMF_OK && v && v < AMF_FULL_VERSION) r = init(v, &factory);
		}
		if (r != AMF_OK) return fail(r, "AMFInit");

		// AMF needs a D3D11 device on an AMD adapter; with several GPUs installed the default adapter is not necessarily one.
		ComPtr<IDXGIFactory1> dxgi;
		CreateDXGIFactory1(IID_PPV_ARGS(&dxgi));
		ComPtr<IDXGIAdapter1> adapter;
		int seen = 0;
		for (UINT i = 0; dxgi && dxgi->EnumAdapters1(i, &adapter) == S_OK; i++, adapter.Reset()) {
			DXGI_ADAPTER_DESC1 d;
			adapter->GetDesc1(&d);
			if (d.VendorId == 0x1002 && seen++ == gpu) { memcpy(&adapter_luid, &d.AdapterLuid, sizeof adapter_luid); break; }
		}
		if (!adapter) { fprintf(stderr, "amf: no AMD adapter with index %d\n", gpu); return false; }
		if (FAILED(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, nullptr, nullptr))) {
			fprintf(stderr, "amf: cannot create D3D11 device on the AMD adapter\n");
			return false;
		}
		if ((r = factory->CreateContext(&ctx)) != AMF_OK) return fail(r, "CreateContext");
		if ((r = ctx->InitDX11(device.Get())) != AMF_OK) return fail(r, "InitDX11");
		if ((r = factory->CreateComponent(ctx, n->component, &enc)) != AMF_OK) return fail(r, "CreateComponent");

		enc->SetProperty(n->usage, n->usage_transcoding);
		enc->SetProperty(n->rc, n->rc_cbr);
		enc->SetProperty(n->target, (amf_int64)p.bitrate);
		enc->SetProperty(n->peak, (amf_int64)p.bitrate);
		enc->SetProperty(n->rate, AMFConstructRate(p.framerate.num, p.framerate.den));
		enc->SetProperty(n->gop, (amf_int64)p.gop);
		enc->SetProperty(n->bitdepth, (amf_int64)(depth > 8 ? 10 : 8));
		// The runtime does not pick the profile from the bit depth: a 10-bit HEVC stream said Main (ffprobe: profile=Main, yuv420p10le), which NVIDIA's Vulkan
		// decoder refuses. Main 10 unless the options say otherwise (they are set after this).
		if (p.codec == "hevc" && depth > 8) enc->SetProperty(AMF_VIDEO_ENCODER_HEVC_PROFILE, (amf_int64)AMF_VIDEO_ENCODER_HEVC_PROFILE_MAIN_10);
		set_color(p.color);
		NativeOptions native_opts;
		std::string why;
		if (!resolve_options("amf", p.codec, o, [](const std::string& k) { return k; }, native_opts, why)) { fprintf(stderr, "amf: %s\n", why.c_str()); return false; }
		if (p.codec == "av1" && std::none_of(native_opts.begin(), native_opts.end(), [](const auto& kv) { return kv.first == "alignment_mode"; })) {
			// AV1 is coded in 64x16 blocks only [H Av1AlignmentMode, its default], which AMD recommends for quality over the modes that code other sizes
			// (RDNA 3 refuses 1080 lines in this mode, so 1920x1080 is coded as 1920x1088). The picture is padded to that by repeating its edges, and
			// the padding is cropped off in the container (coded_w/coded_h), since AV1 itself cannot crop.
			cw = (w + 63) / 64 * 64; ch = (h + 15) / 16 * 16;
			if (enc->SetProperty(AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE, (amf_int64)AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_64X16_ONLY) != AMF_OK) cw = w, ch = h; // a runtime without the property
		}
		if (cw != w || ch != h) { coded_w = cw; coded_h = ch; }
		enc->SetProperty(n->size, AMFConstructSize(cw, ch));

		// The pre-encode filter is a component of its own in front of the encoder; its options (pre_filter, pre_filter_strength, pre_filter_sensitivity) are not encoder properties.
		std::map<std::string, std::string> filter_opts;
		for (auto it = native_opts.begin(); it != native_opts.end();) {
			if (it->first.compare(0, 10, "pre_filter") == 0) { filter_opts.emplace(it->first, it->second); it = native_opts.erase(it); }
			else ++it;
		}

		// Pre-analysis adds its PA* properties to the encoder only once it is on, so those are looked up after it was set.
		std::map<std::string, std::string> names;
		collect_names(names);
		names_before_pa = names;
		auto apply = [&](const std::string& k, const std::string& v) {
			auto it = names.find(k);
			if (it == names.end()) {
				std::string known;
				for (auto& [n, native] : names) known += (known.empty() ? "" : ", ") + n;
				fprintf(stderr, "amf: unknown property '%s'; this encoder has: %s\n", k.c_str(), known.c_str());
				return false;
			}
			return set(it->second, v);
		};
		for (auto& [k, v] : native_opts) if (k == "enable_pre_analysis" && !apply(k, v)) return false;
		collect_names(names);
		names_all = names;
		for (auto& [k, v] : native_opts) if (k != "enable_pre_analysis" && !apply(k, v)) return false;

		// What this encoder is about to run with, before the driver gets it: when an option takes the whole system down (static scene detection
		// bluescreened the 780M), this is the last line in the log and names the culprit.
		{
			std::string line;
			for (auto& [k, v] : native_opts) line += " " + k + "=" + v;
			for (auto& [k, v] : filter_opts) line += " " + k + "=" + v;
			if (cw != w || ch != h) fprintf(stderr, "amf: opening %s %dx%d (coded %dx%d) with%s\n", p.codec.c_str(), w, h, cw, ch, line.empty() ? " defaults" : line.c_str());
			else fprintf(stderr, "amf: opening %s %dx%d with%s\n", p.codec.c_str(), w, h, line.empty() ? " defaults" : line.c_str());
		}
		const auto surface = depth > 8 ? amf::AMF_SURFACE_P010 : amf::AMF_SURFACE_NV12;
		if ((r = enc->Init(surface, cw, ch)) != AMF_OK) return fail(r, "Init");
		read_caps();
		if (!filter_opts.empty() && !open_filter(filter_opts, p)) return false;
		amf::AMFVariant bp;
		if (enc->GetProperty(L"BPicturesPattern", &bp) == AMF_OK && bp.ToInt64() > 0) b_delay = (int)bp.ToInt64() + 1; // H.264; the other codecs have no such property
		rx = std::thread([this] { receive(); });
		return true;
	}

	int reorder_delay() const override { return b_delay; }

	bool encode(const Picture* fr) override {
		AMF_RESULT r;
		if (!fr) { // flush
			if ((r = enc->Drain()) != AMF_OK) return fail(r, "Drain");
			for (int i = 0; !eof && i < 20000; i++) std::this_thread::sleep_for(std::chrono::milliseconds(1));
			return eof;
		}
		amf::AMFSurfacePtr surf;
		Picture mem; // a frame on the GPU read back, when it cannot be taken from VRAM
#ifndef VES_NO_VULKAN
		if (fr->gpu && gstate >= 0) {
			const VkDecodedFrame* bf = (const VkDecodedFrame*)fr->gpu.get();
			if (bf->buffer && (gstate == 1 || gpu_open(*bf)) && !(surf = gpu_surface(*bf))) return false;
		}
#endif
		if (!surf) {
			if (fr->gpu) {
				mem = *fr;
				if (!picture_to_memory(mem)) return false;
				fr = &mem;
			}
			if ((r = ctx->AllocSurface(amf::AMF_MEMORY_HOST, depth > 8 ? amf::AMF_SURFACE_P010 : amf::AMF_SURFACE_NV12, cw, ch, &surf)) != AMF_OK) return fail(r, "AllocSurface");
			repack(fr, surf);
		}
		surf->SetPts(fr->pts);
		if ((r = surf->Convert(amf::AMF_MEMORY_DX11)) != AMF_OK) return fail(r, "Convert");

		amf::AMFDataPtr data(surf);
		if (filter_pp && !filter(surf, data)) return false;
		for (;;) {
			r = enc->SubmitInput(data);
			if (r != AMF_INPUT_FULL) break;
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		if (r != AMF_OK) return fail(r, "SubmitInput");
		return true;
	}
};

} // namespace

std::unique_ptr<Encoder> make_encoder_amf() { return std::make_unique<Amf>(); }

void amf_dedicate_process(int gpu) { dedicated_gpu = gpu; }
std::string amf_runtime() { return runtime_used; }

// Whether this machine's AMD GPU can encode the codec at that bit depth, found by initializing a throwaway encoder once.
bool amf_supports(const std::string& codec, int chroma, int depth, int gpu) {
	if (chroma != 420 || depth > 10) return false;
	static std::mutex m;
	static std::map<std::string, bool> cache;
	std::lock_guard<std::mutex> lock(m);
	const std::string key = codec + std::to_string(depth) + "/" + std::to_string(gpu);
	auto it = cache.find(key);
	if (it == cache.end()) {
		Params p;
		p.width = 1280; p.height = 720; p.bitrate = 5000000; p.gop = 60;
		p.pix = planar_fmt(420, depth > 8 ? 10 : 8);
		p.codec = codec;
		p.options["gpu"] = std::to_string(gpu);
		probing = true;
		const bool ok = make_encoder_amf()->open(p, nullptr);
		probing = false;
		it = cache.emplace(key, ok).first;
	}
	return it->second;
}

// The capabilities of the codec on AMD GPU `gpu` (see Amf::read_caps), "null" when it cannot encode it. Asked once per codec and GPU.
std::string amf_caps_json(const std::string& codec, int gpu) {
	static std::mutex m;
	static std::map<std::string, std::string> cache;
	std::lock_guard<std::mutex> lock(m);
	const std::string key = codec + "/" + std::to_string(gpu);
	auto it = cache.find(key);
	if (it == cache.end()) {
		Params p;
		p.width = 1280; p.height = 720; p.bitrate = 5000000; p.gop = 60;
		p.pix = planar_fmt(420, 8);
		p.codec = codec;
		p.options["gpu"] = std::to_string(gpu);
		probing = true;
		Amf probe;
		const bool ok = probe.open(p, nullptr);
		probing = false;
		it = cache.emplace(key, ok ? probe.caps() : "null").first;
	}
	return it->second;
}

// What `ves --probe-amf` answers: whether AMD GPU `gpu` can encode the codec and its capabilities, from one encoder opened without hiding why it fails (the reasons go to
// stderr, which the worker shows when the GPU is not offered).
bool amf_probe(const std::string& codec, int gpu, std::string& caps) {
	Params p;
	p.width = 1280; p.height = 720; p.bitrate = 5000000; p.gop = 60;
	p.pix = planar_fmt(420, 8);
	p.codec = codec;
	p.options["gpu"] = std::to_string(gpu);
	// Never closed: the process ends right after answering (exit_now), and AMD's runtime is not trusted to wind down before the answer is out.
	auto* probe = new Amf;
	if (!probe->open(p, nullptr)) return false;
	caps = probe->caps();
	return true;
}

// What `ves --list-options` shows for AMF: the properties of an encoder opened on AMD GPU `gpu`, asked with pre-analysis on (which adds its
// properties) if the GPU can do that. Never closed, as in amf_probe: the process ends after the list.
bool amf_options(const std::string& codec, int gpu, OptionList& out, std::string& why) {
	Params p;
	p.width = 1280; p.height = 720; p.bitrate = 5000000; p.gop = 60;
	p.pix = planar_fmt(420, 8);
	p.codec = codec;
	p.options["gpu"] = std::to_string(gpu);
	p.options["enable_pre_analysis"] = "true";
	probing = true;
	auto* probe = new Amf;
	bool ok = probe->open(p, nullptr);
	if (!ok) {
		p.options.erase("enable_pre_analysis");
		probe = new Amf;
		ok = probe->open(p, nullptr);
	}
	probing = false;
	if (!ok) { why = "AMD GPU " + std::to_string(gpu) + " cannot open an AMF " + codec + " encoder"; return false; }
	out = probe->options();
	return true;
}

#else
// AMF's Linux runtime needs a Vulkan context (AMFContext::InitVulkan); not implemented, so the encoder reports itself as unavailable.
#include <cstdio>

#include "options.h"

namespace {
struct AmfUnavailable : Encoder {
	bool open(const Params&, Sink) override { fprintf(stderr, "amf: not supported on this platform\n"); return false; }
	bool encode(const Picture*) override { return false; }
};
} // namespace

std::unique_ptr<Encoder> make_encoder_amf() { return std::make_unique<AmfUnavailable>(); }
bool amf_supports(const std::string&, int, int, int) { return false; }
std::string amf_caps_json(const std::string&, int) { return "null"; }
bool amf_probe(const std::string&, int, std::string&) { fprintf(stderr, "amf: not supported on this platform\n"); return false; }
void amf_dedicate_process(int) {}
std::string amf_runtime() { return {}; }
bool amf_options(const std::string&, int, OptionList&, std::string& why) { why = "AMF is not supported on this platform"; return false; }
#endif
