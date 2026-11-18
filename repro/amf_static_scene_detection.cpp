// Reproduces a Windows bugcheck 0x119 (VIDEO_SCHEDULER_INTERNAL_ERROR) in amdkmdag.sys, raised through watchdog.sys, on a Radeon 780M (RDNA 3):
// two or more AMF encoders run at the same time, each with pre-analysis and static scene detection (PAStaticSceneDetectionEnable) on with
// sensitivity 0 or 1 (with the default sensitivity it does nothing, and nothing crashes). Seen with H.264 and HEVC; AV1 ignores the setting.
// The same settings in a single encoder work, unless PAMaxQPBeforeForceSkip is set as well, which crashes even with one.
//
// Uses only the public AMF API (amfrt64.dll from the driver, loaded at run time) and D3D11. Every encoder gets a D3D11 device of its own on the first
// AMD adapter, gets NV12 pictures made on the CPU (a moving pattern, then the same picture again and again, so static scene detection has something to
// skip), and encodes them at 1920x1080, 60 fps, CBR 6 Mbps, usage transcoding, preset balanced. Nothing is written to disk but this program's output.
//
// Build (Windows; the AMF SDK's amf/public/include folder is the only dependency, https://github.com/GPUOpen-LibrariesAndSDKs/AMF):
//   clang++ -std=c++17 -O2 -I <AMF>/amf/public/include amf_static_scene_detection.cpp -o amf_static_scene_detection.exe -ld3d11 -ldxgi -lole32
//   (MinGW: x86_64-w64-mingw32-g++ the same way, plus -static)
// Run:
//   amf_static_scene_detection.exe [options]
//     --codec h264|hevc|av1   (default hevc; AV1 ignores static scene detection on the 780M and did not crash)
//     --sessions N            encoders at the same time (default 2)
//     --ssd 0|1               static scene detection (default 1)
//     --sensitivity 0|1|2|-1  PAStaticSceneDetectionSensitivity; -1 leaves it at the default, high (default 0)
//     --maxqp N|-1            PAMaxQPBeforeForceSkip; -1 leaves it unset (default -1)
//     --depth N               PALookAheadBufferDepth (default 11)
//     --frames N              pictures per encoder (default 600: 10 s, alternating 1 s of motion and 1.4 s of a still picture)
//   amf_static_scene_detection.exe --sessions 2               bluescreens the 780M within seconds
//   amf_static_scene_detection.exe --sessions 1               encodes
//   amf_static_scene_detection.exe --sessions 1 --maxqp 20    bluescreens the 780M
//   amf_static_scene_detection.exe --sessions 2 --ssd 0       encodes
//
// Observed with ves (https://github.com/Xaymar/video-encoder-sampler), which does the same through the same calls, 2026-11-18:
//   Radeon 780M (RDNA 3), AMF runtime 1.5.3, latest driver for it: 2+ encoders with static scene detection: bugcheck 0x119 every time, after the
//   encoders slowed down to about 1.3 fps together; 1 encoder: works, and sensitivity 0/1 save ~10% bitrate on still parts; 1 encoder with
//   PAMaxQPBeforeForceSkip 20: bugcheck 0x119.
//   Ryzen 7 9800X3D integrated GPU (RDNA 2), AMF runtime 1.4.37: 4 encoders with the same settings work.
// Expected: the encoders work (or Init fails), and the GPU never takes Windows down.
//
// Every line is printed and flushed before the call it names, so after a bluescreen the last lines of a log (amf_static_scene_detection.exe > log.txt)
// show where it was. Windows may lose the last lines if they were not yet on the disk; running it from a console and taking a photo also works.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi.h>

#include <core/Factory.h>
#include <components/VideoEncoderVCE.h>
#include <components/VideoEncoderHEVC.h>
#include <components/VideoEncoderAV1.h>
#include <components/PreAnalysis.h>

#include <atomic>
#include <cstdarg>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef _MSC_VER
#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "ole32.lib")
#endif

struct Options {
	std::string codec = "hevc";
	int sessions = 2, ssd = 1, sensitivity = 0, maxqp = -1, depth = 11, frames = 600;
};

static const auto t0 = std::chrono::steady_clock::now();
static std::mutex out_m;
static void say(const char* fmt, ...) {
	std::lock_guard<std::mutex> lock(out_m);
	printf("[%8.3f] ", std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
	va_list a;
	va_start(a, fmt);
	vprintf(fmt, a);
	va_end(a);
	printf("\n");
	fflush(stdout);
}

static amf::AMFFactory* factory = nullptr;

// A D3D11 device on the first AMD adapter (vendor 0x1002): on a laptop the default adapter may be another GPU.
static ID3D11Device* amd_device(std::string& name) {
	IDXGIFactory1* f = nullptr;
	if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&f))) return nullptr;
	ID3D11Device* dev = nullptr;
	IDXGIAdapter1* a = nullptr;
	for (UINT i = 0; !dev && f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; i++) {
		DXGI_ADAPTER_DESC1 d{};
		a->GetDesc1(&d);
		if (d.VendorId == 0x1002) {
			char n[128] = {};
			WideCharToMultiByte(CP_UTF8, 0, d.Description, -1, n, sizeof n - 1, nullptr, nullptr);
			name = n;
			const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
			if (FAILED(D3D11CreateDevice(a, D3D_DRIVER_TYPE_UNKNOWN, nullptr, D3D11_CREATE_DEVICE_VIDEO_SUPPORT, levels, 2, D3D11_SDK_VERSION, &dev, nullptr, nullptr)))
				dev = nullptr;
		}
		a->Release();
	}
	f->Release();
	return dev;
}

// Picture n: 60 pictures of a moving pattern, then 84 of the last one again, and so on.
static void fill(amf::AMFSurface* s, int n, int w, int h) {
	const int k = n % 144, t = k < 60 ? k : 59;
	amf::AMFPlane* y = s->GetPlaneAt(0);
	amf::AMFPlane* uv = s->GetPlaneAt(1);
	uint8_t* py = (uint8_t*)y->GetNative();
	uint8_t* puv = (uint8_t*)uv->GetNative();
	const int sy = y->GetHPitch(), suv = uv->GetHPitch();
	for (int r = 0; r < h; r++)
		for (int c = 0; c < w; c++) py[(size_t)r * sy + c] = (uint8_t)(((c + t * 8) ^ (r + t * 3)) + ((c * 7 + r * 13 + t * 31) % 17));
	for (int r = 0; r < h / 2; r++)
		for (int c = 0; c < w / 2; c++) {
			puv[(size_t)r * suv + c * 2] = (uint8_t)(128 + ((c + t * 2) % 64) - 32);
			puv[(size_t)r * suv + c * 2 + 1] = (uint8_t)(128 + ((r + t) % 64) - 32);
		}
}

static std::atomic<int> failures{0};

static void session(int id, const Options& o) {
	const int w = 1920, h = 1080;
	const bool hevc = o.codec == "hevc", av1 = o.codec == "av1";
	auto fail = [&](const char* what, AMF_RESULT r) { say("session %d: %s failed: result %d", id, what, (int)r); failures++; };

	std::string adapter;
	say("session %d: creating a D3D11 device", id);
	ID3D11Device* dev = amd_device(adapter);
	if (!dev) { say("session %d: no AMD adapter with D3D11", id); failures++; return; }
	say("session %d: device on %s", id, adapter.c_str());

	amf::AMFContextPtr ctx;
	AMF_RESULT r = factory->CreateContext(&ctx);
	if (r != AMF_OK) return fail("CreateContext", r);
	if ((r = ctx->InitDX11(dev)) != AMF_OK) return fail("InitDX11", r);
	amf::AMFComponentPtr enc;
	if ((r = factory->CreateComponent(ctx, av1 ? AMFVideoEncoder_AV1 : hevc ? AMFVideoEncoder_HEVC : AMFVideoEncoderVCE_AVC, &enc)) != AMF_OK) return fail("CreateComponent", r);

	auto set = [&](const wchar_t* name, const amf::AMFVariant& v) {
		const AMF_RESULT sr = enc->SetProperty(name, v);
		if (sr != AMF_OK) say("session %d: SetProperty(%ls) = result %d", id, name, (int)sr);
	};
	if (av1) {
		set(AMF_VIDEO_ENCODER_AV1_USAGE, amf::AMFVariant((amf_int64)AMF_VIDEO_ENCODER_AV1_USAGE_TRANSCODING));
		set(AMF_VIDEO_ENCODER_AV1_FRAMESIZE, amf::AMFVariant(AMFConstructSize(w, h)));
		set(AMF_VIDEO_ENCODER_AV1_FRAMERATE, amf::AMFVariant(AMFConstructRate(60, 1)));
		set(AMF_VIDEO_ENCODER_AV1_GOP_SIZE, amf::AMFVariant((amf_int64)120));
		set(AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET, amf::AMFVariant((amf_int64)AMF_VIDEO_ENCODER_AV1_QUALITY_PRESET_BALANCED));
		set(AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD, amf::AMFVariant((amf_int64)AMF_VIDEO_ENCODER_AV1_RATE_CONTROL_METHOD_CBR));
		set(AMF_VIDEO_ENCODER_AV1_TARGET_BITRATE, amf::AMFVariant((amf_int64)6000000));
		set(AMF_VIDEO_ENCODER_AV1_PEAK_BITRATE, amf::AMFVariant((amf_int64)6000000));
		set(AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE, amf::AMFVariant((amf_int64)AMF_VIDEO_ENCODER_AV1_ALIGNMENT_MODE_NO_RESTRICTIONS));   // RDNA 3 refuses 1080 lines otherwise
		set(AMF_VIDEO_ENCODER_AV1_PRE_ANALYSIS_ENABLE, amf::AMFVariant(true));
	} else if (hevc) {
		set(AMF_VIDEO_ENCODER_HEVC_USAGE, amf::AMFVariant((amf_int64)AMF_VIDEO_ENCODER_HEVC_USAGE_TRANSCODING));
		set(AMF_VIDEO_ENCODER_HEVC_FRAMESIZE, amf::AMFVariant(AMFConstructSize(w, h)));
		set(AMF_VIDEO_ENCODER_HEVC_FRAMERATE, amf::AMFVariant(AMFConstructRate(60, 1)));
		set(AMF_VIDEO_ENCODER_HEVC_GOP_SIZE, amf::AMFVariant((amf_int64)120));
		set(AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET, amf::AMFVariant((amf_int64)AMF_VIDEO_ENCODER_HEVC_QUALITY_PRESET_BALANCED));
		set(AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD, amf::AMFVariant((amf_int64)AMF_VIDEO_ENCODER_HEVC_RATE_CONTROL_METHOD_CBR));
		set(AMF_VIDEO_ENCODER_HEVC_TARGET_BITRATE, amf::AMFVariant((amf_int64)6000000));
		set(AMF_VIDEO_ENCODER_HEVC_PEAK_BITRATE, amf::AMFVariant((amf_int64)6000000));
		set(AMF_VIDEO_ENCODER_HEVC_PRE_ANALYSIS_ENABLE, amf::AMFVariant(true));
	} else {
		set(AMF_VIDEO_ENCODER_USAGE, amf::AMFVariant((amf_int64)AMF_VIDEO_ENCODER_USAGE_TRANSCODING));
		set(AMF_VIDEO_ENCODER_FRAMESIZE, amf::AMFVariant(AMFConstructSize(w, h)));
		set(AMF_VIDEO_ENCODER_FRAMERATE, amf::AMFVariant(AMFConstructRate(60, 1)));
		set(AMF_VIDEO_ENCODER_IDR_PERIOD, amf::AMFVariant((amf_int64)120));
		set(AMF_VIDEO_ENCODER_QUALITY_PRESET, amf::AMFVariant((amf_int64)AMF_VIDEO_ENCODER_QUALITY_PRESET_BALANCED));
		set(AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD, amf::AMFVariant((amf_int64)AMF_VIDEO_ENCODER_RATE_CONTROL_METHOD_CBR));
		set(AMF_VIDEO_ENCODER_TARGET_BITRATE, amf::AMFVariant((amf_int64)6000000));
		set(AMF_VIDEO_ENCODER_PEAK_BITRATE, amf::AMFVariant((amf_int64)6000000));
		set(AMF_VIDEO_ENCODER_PRE_ANALYSIS_ENABLE, amf::AMFVariant(true));
	}
	// The pre-analysis properties exist on the encoder once pre-analysis is on.
	set(AMF_PA_LOOKAHEAD_BUFFER_DEPTH, amf::AMFVariant((amf_int64)o.depth));
	if (o.ssd) {
		set(AMF_PA_STATIC_SCENE_DETECTION_ENABLE, amf::AMFVariant(true));
		if (o.sensitivity >= 0) set(AMF_PA_STATIC_SCENE_DETECTION_SENSITIVITY, amf::AMFVariant((amf_int64)o.sensitivity));
		if (o.maxqp >= 0) set(AMF_PA_MAX_QP_BEFORE_FORCE_SKIP, amf::AMFVariant((amf_int64)o.maxqp));
	}

	say("session %d: Init %s %dx%d, CBR 6 Mbps, PA lookahead %d, static scene detection %d, sensitivity %d, max QP before force skip %d", id,
	    o.codec.c_str(), w, h, o.depth, o.ssd, o.sensitivity, o.maxqp);
	if ((r = enc->Init(amf::AMF_SURFACE_NV12, w, h)) != AMF_OK) return fail("Init", r);
	say("session %d: Init done", id);

	size_t bytes = 0;
	int out = 0, n = 0;
	auto drain = [&](bool wait) {
		for (;;) {
			amf::AMFDataPtr d;
			const AMF_RESULT qr = enc->QueryOutput(&d);
			if (qr == AMF_OK && d) {
				amf::AMFBufferPtr b(d);
				if (b) bytes += b->GetSize();
				if (++out % 60 == 0) say("session %d: %d pictures out, %.0f kbps so far", id, out, bytes * 8.0 * 60 / out / 1000);
				continue;
			}
			if (qr == AMF_EOF) return true;
			if (qr != AMF_OK && qr != AMF_REPEAT) { fail("QueryOutput", qr); return true; }
			if (!wait) return false;
			Sleep(1);
		}
	};
	while (n < o.frames) {
		amf::AMFSurfacePtr host;
		if ((r = ctx->AllocSurface(amf::AMF_MEMORY_HOST, amf::AMF_SURFACE_NV12, w, h, &host)) != AMF_OK) return fail("AllocSurface", r);
		fill(host, n, w, h);
		if ((r = host->Convert(amf::AMF_MEMORY_DX11)) != AMF_OK) return fail("Convert(DX11)", r);
		host->SetPts((amf_pts)n * AMF_SECOND / 60);
		if (n % 60 == 0) say("session %d: submitting picture %d (%s)", id, n, n % 144 < 60 ? "motion" : "still");
		for (;;) {
			r = enc->SubmitInput(host);
			if (r != AMF_INPUT_FULL) break;
			drain(false);
			Sleep(1);
		}
		if (r != AMF_OK) return fail("SubmitInput", r);
		n++;
		drain(false);
	}
	say("session %d: draining", id);
	enc->Drain();
	drain(true);
	say("session %d: done, %d pictures, %.0f kbps", id, out, out ? bytes * 8.0 * 60 / out / 1000 : 0.0);
	enc->Terminate();
	ctx->Terminate();
	dev->Release();
}

int main(int argc, char** argv) {
	Options o;
	for (int i = 1; i + 1 < argc; i += 2) {
		const std::string k = argv[i], v = argv[i + 1];
		if (k == "--codec") o.codec = v;
		else if (k == "--sessions") o.sessions = atoi(v.c_str());
		else if (k == "--ssd") o.ssd = atoi(v.c_str());
		else if (k == "--sensitivity") o.sensitivity = atoi(v.c_str());
		else if (k == "--maxqp") o.maxqp = atoi(v.c_str());
		else if (k == "--depth") o.depth = atoi(v.c_str());
		else if (k == "--frames") o.frames = atoi(v.c_str());
		else { printf("unknown option %s (see the comment at the top of the source)\n", k.c_str()); return 2; }
	}
	if (o.codec != "hevc" && o.codec != "h264" && o.codec != "av1") { printf("--codec is h264, hevc or av1\n"); return 2; }

	HMODULE dll = LoadLibraryW(AMF_DLL_NAME);
	if (!dll) { printf("cannot load %ls\n", AMF_DLL_NAME); return 1; }
	auto init = (AMFInit_Fn)GetProcAddress(dll, AMF_INIT_FUNCTION_NAME);
	auto version = (AMFQueryVersion_Fn)GetProcAddress(dll, AMF_QUERY_VERSION_FUNCTION_NAME);
	amf_uint64 v = 0;
	if (version) version(&v);
	if (!init || init(AMF_FULL_VERSION, &factory) != AMF_OK) { printf("AMFInit failed\n"); return 1; }
	say("AMF runtime %d.%d.%d, %s, %d session(s)", (int)AMF_GET_MAJOR_VERSION(v), (int)AMF_GET_MINOR_VERSION(v), (int)AMF_GET_SUBMINOR_VERSION(v),
	    o.codec.c_str(), o.sessions);

	std::vector<std::thread> threads;
	for (int i = 0; i < o.sessions; i++) threads.emplace_back(session, i, std::cref(o));
	for (auto& t : threads) t.join();
	say(failures ? "finished with %d failure(s)" : "finished, no crash", (int)failures);
	return failures ? 1 : 0;
}
