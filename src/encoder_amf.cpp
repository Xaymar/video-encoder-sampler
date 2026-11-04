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

#include <components/PreProcessing.h>
#include <components/VideoEncoderAV1.h>
#include <components/VideoEncoderHEVC.h>
#include <components/VideoEncoderVCE.h>
#include <core/Factory.h>

using Microsoft::WRL::ComPtr;

namespace {

thread_local bool probing = false; // set while amf_supports() tries a codec out

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
			if (native.find("HalfPixel") != std::string::npos || native.find("QuarterPixel") != std::string::npos || native == "AdaptiveMiniGOP") continue; // GCN4 and older only, which AMD no longer supports
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
			if (name == "rate_control_preanalysis_enable" || name == "rate_control_pre_analysis_enable") name = "pre_encode"; // pre-encode assisted rate control
			names.emplace(name, native);
		}
	}

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
		filter_pp->SetProperty(AMF_PP_ADAPTIVE_FILTER_ENABLE, true);
		filter_pp->SetProperty(AMF_PP_ADAPTIVE_FILTER_STRENGTH, strength);
		filter_pp->SetProperty(AMF_PP_ADAPTIVE_FILTER_SENSITIVITY, sensitivity);
		filter_pp->SetProperty(AMF_PP_TARGET_BITRATE, (amf_int64)p.bitrate);
		filter_pp->SetProperty(AMF_PP_FRAME_RATE, AMFConstructRate(p.framerate.num, p.framerate.den));
		if ((r = filter_pp->Init(depth > 8 ? amf::AMF_SURFACE_P010 : amf::AMF_SURFACE_NV12, w, h)) != AMF_OK) return fail(r, "Init(PreProcessing)");
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
		set_color(p.color);
		NativeOptions native_opts;
		std::string why;
		if (!resolve_options("amf", p.codec, o, [](const std::string& k) { return k; }, native_opts, why)) { fprintf(stderr, "amf: %s\n", why.c_str()); return false; }

		// The pre-encode filter is a component of its own in front of the encoder; its options (pre_filter, pre_filter_strength, pre_filter_sensitivity) are not encoder properties.
		std::map<std::string, std::string> filter_opts;
		for (auto it = native_opts.begin(); it != native_opts.end();) {
			if (it->first.compare(0, 10, "pre_filter") == 0) { filter_opts.emplace(it->first, it->second); it = native_opts.erase(it); }
			else ++it;
		}

		// Pre-analysis adds its PA* properties to the encoder only once it is on, so those are looked up after it was set.
		std::map<std::string, std::string> names;
		collect_names(names);
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
		for (auto& [k, v] : native_opts) if (k != "enable_pre_analysis" && !apply(k, v)) return false;

		if ((r = enc->Init(depth > 8 ? amf::AMF_SURFACE_P010 : amf::AMF_SURFACE_NV12, w, h)) != AMF_OK) return fail(r, "Init");
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
		if ((r = ctx->AllocSurface(amf::AMF_MEMORY_HOST, depth > 8 ? amf::AMF_SURFACE_P010 : amf::AMF_SURFACE_NV12, w, h, &surf)) != AMF_OK) return fail(r, "AllocSurface");
		repack(fr, surf);
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

#else
// AMF's Linux runtime needs a Vulkan context (AMFContext::InitVulkan); not implemented, so the encoder reports itself as unavailable.
#include <cstdio>

namespace {
struct AmfUnavailable : Encoder {
	bool open(const Params&, Sink) override { fprintf(stderr, "amf: not supported on this platform\n"); return false; }
	bool encode(const Picture*) override { return false; }
};
} // namespace

std::unique_ptr<Encoder> make_encoder_amf() { return std::make_unique<AmfUnavailable>(); }
bool amf_supports(const std::string&, int, int, int) { return false; }
std::string amf_caps_json(const std::string&, int) { return "null"; }
#endif
