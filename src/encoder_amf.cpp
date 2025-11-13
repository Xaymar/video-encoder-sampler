// Native AMD AMF backend: talks to amfrt64.dll directly (loaded at runtime), no libavcodec.
// Encoder options are raw AMF property names (e.g. QualityPreset, BPicturesPattern for H.264,
// HevcQualityPreset for HEVC) so every property AMF exposes can be swept. Enums are numeric.
#include "encoder.h"

#ifdef _WIN32

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <thread>

#include "formats.h"
#include "repack.h"

#include <d3d11.h>
#include <dxgi.h>
#include <wrl/client.h>

#include <components/VideoEncoderAV1.h>
#include <components/VideoEncoderHEVC.h>
#include <components/VideoEncoderVCE.h>
#include <core/Factory.h>

using Microsoft::WRL::ComPtr;

namespace {

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

class Amf : public Encoder {
	HMODULE dll = nullptr;
	amf::AMFFactory* factory = nullptr;
	ComPtr<ID3D11Device> device;
	amf::AMFContextPtr ctx;
	amf::AMFComponentPtr enc;
	int depth = 8;
	Sink sink;
	int w = 0, h = 0;
	// Finished packets are collected on a thread of their own; AMF has no blocking read, so it polls.
	std::thread rx;
	std::atomic<bool> eof{false}, stop{false};
	int b_delay = 0;

	bool fail(AMF_RESULT r, const char* what) {
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
		case amf::AMF_VARIANT_INT64: v = (amf_int64)strtoll(val.c_str(), nullptr, 10); break;
		case amf::AMF_VARIANT_DOUBLE: v = strtod(val.c_str(), nullptr); break;
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
	~Amf() override {
		stop = true;
		if (rx.joinable()) rx.join();
		if (enc) enc->Terminate();
		enc = nullptr;
		if (ctx) ctx->Terminate();
		ctx = nullptr;
		device.Reset();
		// shortcut: amfrt64.dll stays loaded; unloading it while AMF's worker threads wind down has been unreliable
	}

	bool open(const Params& p, Sink s) override {
		sink = std::move(s);
		w = p.width; h = p.height;
		const Fmt fmt = p.pix;
		depth = fmt.depth;
		if (fmt.chroma != 420 || depth > 10) { fprintf(stderr, "amf: cannot encode %d-bit %d\n", depth, fmt.chroma); return false; }
		const Names* n = p.codec == "h264" ? &H264 : p.codec == "hevc" ? &HEVC : p.codec == "av1" ? &AV1 : nullptr;
		if (!n) { fprintf(stderr, "amf: unknown codec '%s'\n", p.codec.c_str()); return false; }
		Options o = p.options;
		int gpu = 0;
		if (auto it = o.find("gpu"); it != o.end()) { gpu = atoi(it->second.c_str()); o.erase(it); }

		dll = LoadLibraryW(AMF_DLL_NAME);
		auto init = dll ? (AMFInit_Fn)GetProcAddress(dll, AMF_INIT_FUNCTION_NAME) : nullptr;
		if (!init) { fprintf(stderr, "amf: amfrt64.dll not found\n"); return false; }
		AMF_RESULT r = init(AMF_FULL_VERSION, &factory);
		if (r != AMF_OK) return fail(r, "AMFInit");

		// AMF needs a D3D11 device on an AMD adapter; with several GPUs installed the default adapter is not necessarily one.
		ComPtr<IDXGIFactory1> dxgi;
		CreateDXGIFactory1(IID_PPV_ARGS(&dxgi));
		ComPtr<IDXGIAdapter1> adapter;
		int seen = 0;
		for (UINT i = 0; dxgi && dxgi->EnumAdapters1(i, &adapter) == S_OK; i++, adapter.Reset()) {
			DXGI_ADAPTER_DESC1 d;
			adapter->GetDesc1(&d);
			if (d.VendorId == 0x1002 && seen++ == gpu) break;
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
		enc->SetProperty(n->size, AMFConstructSize(w, h));
		enc->SetProperty(n->rate, AMFConstructRate(p.framerate.num, p.framerate.den));
		enc->SetProperty(n->gop, (amf_int64)p.gop);
		enc->SetProperty(n->bitdepth, (amf_int64)(depth > 8 ? 10 : 8));
		for (auto& [k, v] : o) if (!set(k, v)) return false;

		if ((r = enc->Init(depth > 8 ? amf::AMF_SURFACE_P010 : amf::AMF_SURFACE_NV12, w, h)) != AMF_OK) return fail(r, "Init");
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
		if ((r = ctx->AllocSurface(amf::AMF_MEMORY_HOST, depth > 8 ? amf::AMF_SURFACE_P010 : amf::AMF_SURFACE_NV12, w, h, &surf)) != AMF_OK) return fail(r, "AllocSurface");
		repack(fr, surf);
		surf->SetPts(fr->pts);
		if ((r = surf->Convert(amf::AMF_MEMORY_DX11)) != AMF_OK) return fail(r, "Convert");

		for (;;) {
			r = enc->SubmitInput(surf);
			if (r != AMF_INPUT_FULL) break;
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
		}
		if (r != AMF_OK) return fail(r, "SubmitInput");
		return true;
	}
};

} // namespace

std::unique_ptr<Encoder> make_encoder_amf() { return std::make_unique<Amf>(); }

// Whether this machine's AMD GPU can encode the codec at that bit depth, found by initializing a throwaway encoder once.
bool amf_supports(const std::string& codec, int chroma, int depth) {
	if (chroma != 420 || depth > 10) return false;
	static std::mutex m;
	static std::map<std::string, bool> cache;
	std::lock_guard<std::mutex> lock(m);
	const std::string key = codec + std::to_string(depth);
	auto it = cache.find(key);
	if (it == cache.end()) {
		Params p;
		p.width = 1280; p.height = 720; p.bitrate = 5000000; p.gop = 60;
		p.pix = planar_fmt(420, depth > 8 ? 10 : 8);
		p.codec = codec;
		it = cache.emplace(key, make_encoder_amf()->open(p, nullptr)).first;
	}
	return it->second;
}

#else
// AMF's Linux runtime needs a Vulkan context (AMFContext::InitVulkan); not implemented, so the encoder reports itself as unavailable.
#include <cstdio>

namespace {
struct AmfUnavailable : Encoder {
	bool open(const Params&, Sink) override { fprintf(stderr, "amf: not supported on this platform
"); return false; }
	bool encode(const Picture*) override { return false; }
};
} // namespace

std::unique_ptr<Encoder> make_encoder_amf() { return std::make_unique<AmfUnavailable>(); }
bool amf_supports(const std::string&, int, int) { return false; }
#endif
