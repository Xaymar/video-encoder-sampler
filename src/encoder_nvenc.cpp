// Native NVENC backend: talks to nvEncodeAPI directly (loaded at runtime), no libavcodec.
#include "encoder.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "formats.h"
#include "options.h"
#include "gpu.h"
#include "repack.h"
#include "decode.h"
#include "vk_compute.h"

#ifdef _WIN32
#include <windows.h>
constexpr bool ASYNC = true; // the encoder signals an event per finished picture; on Linux NVENC has no asynchronous mode
#else
constexpr bool ASYNC = false;
#endif

extern "C" {
#include <ffnvcodec/dynlink_loader.h>
}

#ifndef _WIN32
#include <unistd.h>
inline bool operator==(const GUID& a, const GUID& b) { return memcmp(&a, &b, sizeof a) == 0; } // the Linux header has no comparison
#endif

namespace {

// CUDA numbers its devices in its own order; find the one that is the same GPU as DXGI's n-th NVIDIA adapter.
bool cuda_device_for(CudaFunctions* cu, int index, CUdevice& dev) {
	auto gpus = list_gpus(VENDOR_NVIDIA);
	int count = 0;
	if (cu->cuDeviceGetCount(&count) || index < 0) return false;
#ifdef _WIN32
	if (index < (int)gpus.size() && cu->cuDeviceGetLuid) {
		for (int i = 0; i < count; i++) {
			CUdevice d;
			char luid[8];
			unsigned mask;
			if (cu->cuDeviceGet(&d, i) || cu->cuDeviceGetLuid(luid, &mask, d)) continue;
			uint64_t v;
			memcpy(&v, luid, 8);
			if (v == gpus[index].luid) { dev = d; return true; }
		}
	}
#endif
	return index < count && !cu->cuDeviceGet(&dev, index); // no LUID match: fall back to CUDA's own numbering
}

GUID codec_guid(const std::string& codec, bool& known) {
	known = true;
	if (codec == "h264") return NV_ENC_CODEC_H264_GUID;
	if (codec == "hevc") return NV_ENC_CODEC_HEVC_GUID;
	if (codec == "av1") return NV_ENC_CODEC_AV1_GUID;
	known = false;
	return GUID{};
}

const std::vector<std::pair<std::string, NV_ENC_TUNING_INFO>>& tunes() {
	static const std::vector<std::pair<std::string, NV_ENC_TUNING_INFO>> t = {{"hq", NV_ENC_TUNING_INFO_HIGH_QUALITY}, {"ll", NV_ENC_TUNING_INFO_LOW_LATENCY},
		{"ull", NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY}, {"lossless", NV_ENC_TUNING_INFO_LOSSLESS},
		{"uhq", NV_ENC_TUNING_INFO_ULTRA_HIGH_QUALITY}}; // uhq: HEVC and AV1 only, Turing and newer (nvEncodeAPI.h, NV_ENC_TUNING_INFO)
	return t;
}

// The profiles by their standard names (see options.h).
const std::map<std::string, const GUID*>& profiles(const std::string& codec) {
	static const std::map<std::string, const GUID*> h264 = {{"baseline", &NV_ENC_H264_PROFILE_BASELINE_GUID}, {"main", &NV_ENC_H264_PROFILE_MAIN_GUID},
		{"high", &NV_ENC_H264_PROFILE_HIGH_GUID}, {"constrained_high", &NV_ENC_H264_PROFILE_CONSTRAINED_HIGH_GUID}, {"high10", &NV_ENC_H264_PROFILE_HIGH_10_GUID},
		{"high422", &NV_ENC_H264_PROFILE_HIGH_422_GUID}, {"high444", &NV_ENC_H264_PROFILE_HIGH_444_GUID}};
	static const std::map<std::string, const GUID*> hevc = {{"main", &NV_ENC_HEVC_PROFILE_MAIN_GUID}, {"main10", &NV_ENC_HEVC_PROFILE_MAIN10_GUID}, {"rext", &NV_ENC_HEVC_PROFILE_FREXT_GUID}};
	static const std::map<std::string, const GUID*> av1 = {{"main", &NV_ENC_AV1_PROFILE_MAIN_GUID}};
	return codec == "h264" ? h264 : codec == "hevc" ? hevc : av1;
}

// The options that are fields of NVENC's own structures, under the field's name (see the SDK's nvEncodeAPI.h), set to a number (or true/false);
// enums are their numeric values. Fields of the codec being used only. A QP option sets all three QPs of NV_ENC_QP. `get` reads a field back
// (what `ves --list-options` finds the range of a field with).
struct Field {
	std::function<void(long long)> set;
	std::function<long long()> get;
	bool is_enum = false;
	std::string native; // the SDK's name ("rcParams.vbvBufferSize")
};
using Fields = std::map<std::string, Field>;

Fields fields_for(const GUID& codec, NV_ENC_CONFIG* cfg, NV_ENC_INITIALIZE_PARAMS* ip) {
	Fields fields;
	NV_ENC_RC_PARAMS* rcp = &cfg->rcParams;
#define F(name, s, f) fields[name] = {[s](long long v) { s->f = (decltype(s->f))v; }, [s] { return (long long)s->f; }, std::is_enum_v<decltype(s->f)>, std::string(where) + #f};
	const char* where = "rcParams.";
#define RC(f) F(snake_case(#f), rcp, f)
	RC(maxBitRate) RC(vbvBufferSize) RC(vbvInitialDelay) RC(disableIadapt) RC(disableBadapt) RC(zeroReorderDelay) RC(enableNonRefP) RC(strictGOPTarget)
	RC(targetQuality) RC(targetQualityLSB) RC(lowDelayKeyFrameScale) RC(yDcQPIndexOffset) RC(uDcQPIndexOffset) RC(vDcQPIndexOffset) RC(multiPass)
	RC(cbQPIndexOffset) RC(crQPIndexOffset) RC(lookaheadLevel)
#undef RC
	auto qp = [](NV_ENC_QP& q, long long v) { q.qpInterP = q.qpInterB = q.qpIntra = (uint32_t)v; };
	fields["const_qp"] = {[=](long long v) { qp(rcp->constQP, v); }, [=] { return (long long)rcp->constQP.qpIntra; }, false, "rcParams.constQP"};
	fields["min_qp"] = {[=](long long v) { qp(rcp->minQP, v); rcp->enableMinQP = 1; }, [=] { return (long long)rcp->minQP.qpIntra; }, false, "rcParams.minQP"};
	fields["max_qp"] = {[=](long long v) { qp(rcp->maxQP, v); rcp->enableMaxQP = 1; }, [=] { return (long long)rcp->maxQP.qpIntra; }, false, "rcParams.maxQP"};
	fields["initial_rc_qp"] = {[=](long long v) { qp(rcp->initialRCQP, v); rcp->enableInitialRCQP = 1; }, [=] { return (long long)rcp->initialRCQP.qpIntra; }, false, "rcParams.initialRCQP"};
	where = "NV_ENC_CONFIG.";
	F("mv_precision", cfg, mvPrecision)
	where = "NV_ENC_INITIALIZE_PARAMS.";
#define IP(f) F(snake_case(#f), ip, f)
	IP(enableWeightedPrediction) IP(splitEncodeMode) IP(enableUniDirectionalB)
#undef IP
#define C(f) F(snake_case(#f), c, f)
	if (codec == NV_ENC_CODEC_H264_GUID) {
		auto* c = &cfg->encodeCodecConfig.h264Config;
		where = "h264Config.";
		C(hierarchicalPFrames) C(hierarchicalBFrames) C(enableFillerDataInsertion) C(level) C(disableDeblockingFilterIDC) C(adaptiveTransformMode) C(bdirectMode)
		C(entropyCodingMode) C(tfLevel) F("max_ref_frames_l0", c, numRefL0) F("max_ref_frames_l1", c, numRefL1)
	} else if (codec == NV_ENC_CODEC_HEVC_GUID) {
		auto* c = &cfg->encodeCodecConfig.hevcConfig;
		where = "hevcConfig.";
		C(level) C(tier) C(minCUSize) C(maxCUSize) C(enableFillerDataInsertion) C(tfLevel) C(disableDeblockingFilterIDC)
		F("max_ref_frames_l0", c, numRefL0) F("max_ref_frames_l1", c, numRefL1)
	} else {
		auto* c = &cfg->encodeCodecConfig.av1Config;
		where = "av1Config.";
		C(level) C(tier) C(minPartSize) C(maxPartSize) C(enableCustomTileConfig) C(numTileColumns) C(numTileRows) C(enableFilmGrainParams) C(tfLevel)
		F("max_ref_frames_forward", c, numFwdRefs) F("max_ref_frames_backward", c, numBwdRefs)
	}
#undef C
#undef F
	return fields;
}

// What the options of NVENC take, where its field type does not say: its enums (nvEncodeAPI.h) and values with a meaning of their own.
const std::map<std::string, std::string>& field_values() {
	static const std::map<std::string, std::string> t = {
		{"multi_pass", "0 off | 1 two passes, quarter resolution first | 2 two passes, full resolution first"},
		{"lookahead_level", "0..3 | 15 auto (needs lookahead_frames above 0)"},
		{"mv_precision", "0 default | 1 full pel | 2 half pel | 3 quarter pel"},
		{"split_encode_mode", "0 auto | 1 forced, strips picked by the driver | 2 two strips | 3 three | 4 four | 15 off"},
		{"adaptive_transform_mode", "0 auto | 1 off | 2 on"},
		{"bdirect_mode", "0 auto | 1 off | 2 temporal | 3 spatial"},
		{"entropy_coding_mode", "0 auto | 1 CABAC | 2 CAVLC"},
		{"tf_level", "0 off | 4 temporal filter on"},
		{"max_ref_frames_l0", "0 auto | 1..7"}, {"max_ref_frames_l1", "0 auto | 1..7"},
		{"max_ref_frames_forward", "0 auto | 1..7"}, {"max_ref_frames_backward", "0 auto | 1..7"},
		{"min_cu_size", "0 auto | 1 8x8 | 2 16x16 | 3 32x32 | 4 64x64"}, {"max_cu_size", "0 auto | 1 8x8 | 2 16x16 | 3 32x32 | 4 64x64"},
		{"min_part_size", "0 auto | 1 4x4 | 2 8x8 | 3 16x16 | 4 32x32 | 5 64x64"}, {"max_part_size", "0 auto | 1 4x4 | 2 8x8 | 3 16x16 | 4 32x32 | 5 64x64"},
		{"const_qp", "0..51 (AV1 0..255)"}, {"min_qp", "0..51 (AV1 0..255), also switches the minimum on"},
		{"max_qp", "0..51 (AV1 0..255), also switches the maximum on"}, {"initial_rc_qp", "0..51 (AV1 0..255), also switches it on"},
		{"target_quality", "0 auto | 1..51 (VBR)"},
		{"tier", "main | high"},
	};
	return t;
}

// What a field holds, found by writing to it and reading it back: a bit field of one bit comes back as 0..1.
std::string field_range(const Field& f) {
	f.set(-1);
	const long long all = f.get();
	if (all == -1) { // signed
		for (int b : {7, 15, 31}) {
			const long long m = (1LL << b) - 1;
			f.set(m + 1);
			if (f.get() != m + 1) return std::to_string(-m - 1) + ".." + std::to_string(m);
		}
		return "number";
	}
	return all > 0 && all < 0xFFFFFFFFLL ? "0.." + std::to_string(all) : "0..";
}

// What the GPU and driver can encode for a codec; asked once by opening a throwaway session.
struct Caps {
	bool b10 = false, c444 = false;
};

// Every capability the SDK headers define; the values are a fingerprint of the encoder block (how many B-frames, engines, what it supports).
#define NVENC_CAPS(X) \
	X(NUM_MAX_BFRAMES) X(SUPPORTED_RATECONTROL_MODES) X(SUPPORT_FIELD_ENCODING) X(SUPPORT_MONOCHROME) X(SUPPORT_FMO) \
	X(SUPPORT_QPELMV) X(SUPPORT_BDIRECT_MODE) X(SUPPORT_CABAC) X(SUPPORT_ADAPTIVE_TRANSFORM) X(SUPPORT_STEREO_MVC) \
	X(NUM_MAX_TEMPORAL_LAYERS) X(SUPPORT_HIERARCHICAL_PFRAMES) X(SUPPORT_HIERARCHICAL_BFRAMES) X(LEVEL_MAX) X(LEVEL_MIN) \
	X(SEPARATE_COLOUR_PLANE) X(WIDTH_MAX) X(HEIGHT_MAX) X(SUPPORT_TEMPORAL_SVC) X(SUPPORT_DYN_RES_CHANGE) \
	X(SUPPORT_DYN_BITRATE_CHANGE) X(SUPPORT_DYN_FORCE_CONSTQP) X(SUPPORT_DYN_RCMODE_CHANGE) X(SUPPORT_SUBFRAME_READBACK) X(SUPPORT_CONSTRAINED_ENCODING) \
	X(SUPPORT_INTRA_REFRESH) X(SUPPORT_CUSTOM_VBV_BUF_SIZE) X(SUPPORT_DYNAMIC_SLICE_MODE) X(SUPPORT_REF_PIC_INVALIDATION) X(PREPROC_SUPPORT) \
	X(ASYNC_ENCODE_SUPPORT) X(MB_NUM_MAX) X(MB_PER_SEC_MAX) X(SUPPORT_YUV444_ENCODE) X(SUPPORT_LOSSLESS_ENCODE) \
	X(SUPPORT_SAO) X(SUPPORT_MEONLY_MODE) X(SUPPORT_LOOKAHEAD) X(SUPPORT_TEMPORAL_AQ) X(SUPPORT_10BIT_ENCODE) \
	X(NUM_MAX_LTR_FRAMES) X(SUPPORT_WEIGHTED_PREDICTION) X(DYNAMIC_QUERY_ENCODER_CAPACITY) X(SUPPORT_BFRAME_REF_MODE) X(SUPPORT_EMPHASIS_LEVEL_MAP) \
	X(WIDTH_MIN) X(HEIGHT_MIN) X(SUPPORT_MULTIPLE_REF_FRAMES) X(SUPPORT_ALPHA_LAYER_ENCODING) X(NUM_ENCODER_ENGINES) \
	X(SINGLE_SLICE_INTRA_REFRESH) X(DISABLE_ENC_STATE_ADVANCE) X(OUTPUT_RECON_SURFACE) X(OUTPUT_BLOCK_STATS) X(OUTPUT_ROW_STATS) \
	X(SUPPORT_TEMPORAL_FILTER) X(SUPPORT_LOOKAHEAD_LEVEL) X(SUPPORT_UNIDIRECTIONAL_B) X(SUPPORT_MVHEVC_ENCODE) X(SUPPORT_YUV422_ENCODE)

Caps probe_caps(const std::string& codec) {
	Caps caps;
	bool known;
	GUID guid = codec_guid(codec, known);
	CudaFunctions* cu = nullptr;
	NvencFunctions* nv = nullptr;
	CUcontext ctx = nullptr;
	void* enc = nullptr;
	NV_ENCODE_API_FUNCTION_LIST f{};
	CUdevice dev;
	if (known && cuda_load_functions(&cu, nullptr) >= 0 && nvenc_load_functions(&nv, nullptr) >= 0 && !cu->cuInit(0) && !cu->cuDeviceGet(&dev, 0) &&
	    !cu->cuCtxCreate(&ctx, 0, dev)) {
		f.version = NV_ENCODE_API_FUNCTION_LIST_VER;
		NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS os{};
		os.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
		os.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
		os.device = ctx;
		os.apiVersion = NVENCAPI_VERSION;
		if (!nv->NvEncodeAPICreateInstance(&f) && !f.nvEncOpenEncodeSessionEx(&os, &enc)) {
			auto query = [&](NV_ENC_CAPS c) {
				NV_ENC_CAPS_PARAM cp{};
				cp.version = NV_ENC_CAPS_PARAM_VER;
				cp.capsToQuery = c;
				int v = 0;
				return !f.nvEncGetEncodeCaps(enc, guid, &cp, &v) && v > 0;
			};
			caps.b10 = query(NV_ENC_CAPS_SUPPORT_10BIT_ENCODE);
			caps.c444 = query(NV_ENC_CAPS_SUPPORT_YUV444_ENCODE);
			f.nvEncDestroyEncoder(enc);
		}
	}
	if (ctx) cu->cuCtxDestroy(ctx);
	nvenc_free_functions(&nv);
	cuda_free_functions(&cu);
	return caps;
}

struct Slot {
	NV_ENC_INPUT_PTR in = nullptr; // made when the slot first takes a frame from memory
	NV_ENC_OUTPUT_PTR out = nullptr;
	void* ev = nullptr; // signaled when the encoder is done with this slot's picture
	NV_ENC_INPUT_PTR mapped = nullptr; // a frame from VRAM: the slot's registered buffer, mapped while the encoder has its picture
};

#ifndef VES_NO_VULKAN
constexpr unsigned CUDA_EXTERNAL_MEMORY_DEDICATED_FLAG = 1; // CUDA_EXTERNAL_MEMORY_DEDICATED, which the header lacks

// Frames straight from VRAM: the converter's output (a buffer frame on the GPU, VkDecodedFrame::buffer) is packed into NVENC's layout by a
// compute shader (enc_pack, the same as repack.h does in memory) into a Vulkan buffer whose memory CUDA imports as linear device memory and
// NVENC has registered (one per slot). Only buffers cross from Vulkan to CUDA, never images: sharing Vulkan NV12 images with D3D11 or CUDA was
// seen to corrupt VRAM. The CPU waits for the packing before the picture goes to the encoder.
struct GpuInput {
	struct Res {
		ExportedBuf eb;
		CUexternalMemory ext = nullptr;
		CUdeviceptr ptr = 0;
		void* reg = nullptr; // NVENC's registration
		Pipe pipe;
		bool has_pipe = false;
	};
	std::unique_ptr<Vk> vk;
	int device = -1;
	Buf stage; // the frame's planes copied out of the converter's buffer (pipelines bind fixed buffers)
	VkDeviceSize stage_size = 0;
	std::vector<Res> res; // per slot
	uint32_t pitch = 0;
	VkDeviceSize size = 0;
	VkDeviceSize plane_off[3]{}; // where NVENC reads each plane (420: Y, UV)
};
#endif

const GUID* const PRESETS[] = {
	&NV_ENC_PRESET_P1_GUID, &NV_ENC_PRESET_P2_GUID, &NV_ENC_PRESET_P3_GUID, &NV_ENC_PRESET_P4_GUID,
	&NV_ENC_PRESET_P5_GUID, &NV_ENC_PRESET_P6_GUID, &NV_ENC_PRESET_P7_GUID,
};

class Nvenc : public Encoder {
	CudaFunctions* cu = nullptr;
	NvencFunctions* nv = nullptr;
	NV_ENCODE_API_FUNCTION_LIST f{};
	CUcontext ctx = nullptr;
	void* enc = nullptr;
	std::vector<Slot> slots;
	// Submitting and receiving run on different threads (asynchronous mode: the encoder signals an event per finished picture).
	// Slots go from free_ to pending_ when submitted and back once the receiver has read their output, strictly in submission order.
	std::mutex m;
	std::condition_variable cv;
	std::vector<Slot*> free_;
	std::deque<Slot*> pending_;
	std::thread rx;
	void* eos_event = nullptr; // for the end-of-stream picture, which has no slot
	bool flushing = false, finished = false, stop = false, rx_ok = true;
	// Some option combinations make NVENC take pictures and never finish one: no event fires, no output comes. Once every slot is submitted the encoder has
	// more pictures than it may hold back (see `delay`), so a wait for a free slot (or for the end of the stream) that sees no output for this long gives up:
	// the run fails instead of holding its wave until the device process's watchdog ends it.
	static constexpr auto NO_OUTPUT = std::chrono::seconds(30);
	std::atomic<int64_t> outputs{0}; // pictures read, by read_slot()
	long long submitted = 0;
	NV_ENC_BUFFER_FORMAT buffer_fmt = NV_ENC_BUFFER_FORMAT_NV12;
	int chroma = 420, depth = 8;
	// What open() handed the encoder, for settings().
	NV_ENC_CONFIG used_cfg{};
	NV_ENC_INITIALIZE_PARAMS used_ip{};
	GUID used_codec{};
	int used_preset = 0;
	NV_ENC_TUNING_INFO used_tune = NV_ENC_TUNING_INFO_HIGH_QUALITY;
	Sink sink;
	int w = 0, h = 0;
	size_t delay = 0;
	int b_delay = 0;
	CUdevice cudev = 0;
#ifndef VES_NO_VULKAN
	std::unique_ptr<GpuInput> gin;
	int gin_state = 0; // 0: not tried yet, 1: frames from VRAM, -1: through memory
#endif

	bool fail(NVENCSTATUS s, const char* what) {
		const char* msg = enc ? f.nvEncGetLastErrorString(enc) : "";
		fprintf(stderr, "nvenc: %s failed (status %d) %s\n", what, (int)s, msg ? msg : "");
		return false;
	}

public:
	~Nvenc() override {
		if (rx.joinable()) {
			{ std::lock_guard lock(m); stop = true; }
			cv.notify_all();
			rx.join();
		}
		if (enc) {
			auto unregister = [&](void* ev) {
				if (!ev) return;
				NV_ENC_EVENT_PARAMS ep{};
				ep.version = NV_ENC_EVENT_PARAMS_VER;
				ep.completionEvent = ev;
#ifdef _WIN32
				f.nvEncUnregisterAsyncEvent(enc, &ep);
				CloseHandle((HANDLE)ev);
#endif
			};
			unregister(eos_event);
			for (auto& s : slots) unregister(s.ev);
			for (auto& s : slots) {
				if (s.mapped) f.nvEncUnmapInputResource(enc, s.mapped);
				if (s.in) f.nvEncDestroyInputBuffer(enc, s.in);
				if (s.out) f.nvEncDestroyBitstreamBuffer(enc, s.out);
			}
#ifndef VES_NO_VULKAN
			if (gin) for (auto& r : gin->res) if (r.reg) f.nvEncUnregisterResource(enc, r.reg);
#endif
			f.nvEncDestroyEncoder(enc);
		}
#ifndef VES_NO_VULKAN
		gin_free();
#endif
		if (ctx) cu->cuCtxDestroy(ctx);
		nvenc_free_functions(&nv);
		cuda_free_functions(&cu);
	}

	bool open(const Params& p, Sink s) override {
		sink = std::move(s);
		w = p.width; h = p.height;
		Options o = p.options;
		auto take = [&](const char* k, std::string& v) {
			auto it = o.find(k);
			if (it == o.end()) return false;
			v = it->second; o.erase(it); return true;
		};
		std::string v;

		if (cuda_load_functions(&cu, nullptr) < 0 || nvenc_load_functions(&nv, nullptr) < 0) {
			fprintf(stderr, "nvenc: CUDA or nvEncodeAPI library not found\n");
			return false;
		}
		int gpu = take("gpu", v) ? atoi(v.c_str()) : 0;
		CUdevice& dev = cudev;
		if (cu->cuInit(0) || !cuda_device_for(cu, gpu, dev) || cu->cuCtxCreate(&ctx, 0, dev)) {
			fprintf(stderr, "nvenc: CUDA init failed for gpu %d\n", gpu);
			return false;
		}
		f.version = NV_ENCODE_API_FUNCTION_LIST_VER;
		NVENCSTATUS st = nv->NvEncodeAPICreateInstance(&f);
		if (st) return fail(st, "NvEncodeAPICreateInstance");

		NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS os{};
		os.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
		os.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
		os.device = ctx;
		os.apiVersion = NVENCAPI_VERSION;
		if ((st = f.nvEncOpenEncodeSessionEx(&os, &enc))) {
			enc = nullptr;
			// The driver's answer when every session of the GPU is taken: INCOMPATIBLE_CLIENT_KEY on GeForce cards (their session limit), OUT_OF_MEMORY on others.
			if (st == NV_ENC_ERR_INCOMPATIBLE_CLIENT_KEY || st == NV_ENC_ERR_OUT_OF_MEMORY) {
				no_session = true;
				fprintf(stderr, "nvenc: no free encode session on gpu %d (status %d): the GPU's session limit is used up, by this sweep or another process\n", gpu, (int)st);
				return false;
			}
			return fail(st, "open session");
		}

		bool known;
		GUID codec = codec_guid(p.codec, known);
		const bool h26x = p.codec != "av1";
		if (!known) { fprintf(stderr, "nvenc: unknown codec '%s'\n", p.codec.c_str()); return false; }

		const Fmt fmt = p.pix;
		chroma = fmt.chroma; depth = fmt.depth;
		buffer_fmt = chroma == 444 ? (depth > 8 ? NV_ENC_BUFFER_FORMAT_YUV444_10BIT : NV_ENC_BUFFER_FORMAT_YUV444)
		                           : (depth > 8 ? NV_ENC_BUFFER_FORMAT_YUV420_10BIT : NV_ENC_BUFFER_FORMAT_NV12);
		if (chroma == 422 || depth > 10) { fprintf(stderr, "nvenc: cannot encode %d-bit %d\n", depth, chroma); return false; }

		int preset = 4;
		if (take("preset", v)) {
			preset = (v.size() == 2 && v[0] == 'p') ? v[1] - '0' : 0;
			if (preset < 1 || preset > 7) { fprintf(stderr, "nvenc: bad preset '%s'\n", v.c_str()); return false; }
		}
		NV_ENC_TUNING_INFO tune = NV_ENC_TUNING_INFO_HIGH_QUALITY;
		if (take("tune", v)) {
			bool found = false;
			for (auto& [name, t] : tunes()) if (v == name) { tune = t; found = true; }
			if (!found) { fprintf(stderr, "nvenc: bad tune '%s'\n", v.c_str()); return false; }
		}

		std::string profile;
		const bool has_profile = take("profile", profile);

		NV_ENC_PRESET_CONFIG pc{};
		pc.version = NV_ENC_PRESET_CONFIG_VER;
		pc.presetCfg.version = NV_ENC_CONFIG_VER;
		if ((st = f.nvEncGetEncodePresetConfigEx(enc, codec, *PRESETS[preset - 1], tune, &pc))) return fail(st, "get preset config");
		NV_ENC_CONFIG cfg = pc.presetCfg;
		if (color_full(p.color) || color_described(p.color)) { // the colour of the pictures, as the stream's VUI (sequence header for AV1) says it
			auto vui = [&](NV_ENC_CONFIG_H264_VUI_PARAMETERS& v) {
				v.videoSignalTypePresentFlag = 1;
				v.videoFormat = NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
				v.videoFullRangeFlag = color_full(p.color);
				if (color_described(p.color)) {
					v.colourDescriptionPresentFlag = 1;
					v.colourPrimaries = (NV_ENC_VUI_COLOR_PRIMARIES)p.color.primaries;
					v.transferCharacteristics = (NV_ENC_VUI_TRANSFER_CHARACTERISTIC)p.color.trc;
					v.colourMatrix = (NV_ENC_VUI_MATRIX_COEFFS)p.color.space;
				}
			};
			if (p.codec == "h264") vui(cfg.encodeCodecConfig.h264Config.h264VUIParameters);
			else if (p.codec == "hevc") vui(cfg.encodeCodecConfig.hevcConfig.hevcVUIParameters);
			else {
				auto& c = cfg.encodeCodecConfig.av1Config;
				c.colorRange = color_full(p.color);
				if (color_described(p.color)) {
					c.colorPrimaries = (NV_ENC_VUI_COLOR_PRIMARIES)p.color.primaries;
					c.transferCharacteristics = (NV_ENC_VUI_TRANSFER_CHARACTERISTIC)p.color.trc;
					c.matrixCoefficients = (NV_ENC_VUI_MATRIX_COEFFS)p.color.space;
				}
			}
		}
		if (has_profile) { // the standard names, see options.h; left alone, the encoder picks the profile that fits the chroma layout and bit depth
			const auto& table = profiles(p.codec);
			auto it = table.find(profile);
			if (it == table.end()) { fprintf(stderr, "nvenc: profile '%s' is not available for nvenc/%s\n", profile.c_str(), p.codec.c_str()); return false; }
			cfg.profileGUID = *it->second;
		}

		cfg.gopLength = p.gop;
		{ // chroma layout and bit depth; the input and output depth are kept equal
			const NV_ENC_BIT_DEPTH bd = depth > 8 ? NV_ENC_BIT_DEPTH_10 : NV_ENC_BIT_DEPTH_8;
			const uint32_t idc = chroma == 444 ? 3 : 1;
			if (p.codec == "h264") { auto& c = cfg.encodeCodecConfig.h264Config; c.chromaFormatIDC = idc; c.inputBitDepth = c.outputBitDepth = bd; }
			else if (p.codec == "hevc") { auto& c = cfg.encodeCodecConfig.hevcConfig; c.chromaFormatIDC = idc; c.inputBitDepth = c.outputBitDepth = bd; }
			else { auto& c = cfg.encodeCodecConfig.av1Config; c.chromaFormatIDC = idc; c.inputBitDepth = c.outputBitDepth = bd; }
		}
		if (h26x) {
			(codec == NV_ENC_CODEC_H264_GUID ? cfg.encodeCodecConfig.h264Config.idrPeriod : cfg.encodeCodecConfig.hevcConfig.idrPeriod) = p.gop;
		}
		NV_ENC_RC_PARAMS& rc = cfg.rcParams;
		rc.rateControlMode = NV_ENC_PARAMS_RC_CBR;
		rc.averageBitRate = rc.maxBitRate = (uint32_t)p.bitrate;

		if (take("rate_control", v)) {
			if (v == "cbr") rc.rateControlMode = NV_ENC_PARAMS_RC_CBR;
			else if (v == "vbr") rc.rateControlMode = NV_ENC_PARAMS_RC_VBR;
			else { fprintf(stderr, "nvenc: bad rate_control '%s'\n", v.c_str()); return false; }
		}
		if (take("max_b_frames", v)) cfg.frameIntervalP = atoi(v.c_str()) + 1;
		if (take("lookahead_frames", v)) { rc.lookaheadDepth = (uint16_t)atoi(v.c_str()); rc.enableLookahead = rc.lookaheadDepth > 0; }
		if (take("spatial_aq", v)) { // 0 switches it off, "auto" on with the driver picking the strength per frame, 1..15 on with that strength
			if (v == "auto") { rc.enableAQ = 1; rc.aqStrength = 0; }
			else {
				const int n = atoi(v.c_str());
				if (n < 0 || n > 15 || (n == 0 && v != "0")) { fprintf(stderr, "nvenc: bad spatial_aq '%s' (0, auto or 1..15)\n", v.c_str()); return false; }
				rc.enableAQ = n > 0;
				rc.aqStrength = n;
			}
		}
		if (take("temporal_aq", v)) rc.enableTemporalAQ = atoi(v.c_str()) != 0;
		if (take("b_reference_mode", v)) { // none | middle | each | hierarchical (NV_ENC_BFRAME_REF_MODE; hierarchical has requirements of its own, see config.all-options.jsonc)
			NV_ENC_BFRAME_REF_MODE m;
			if (v == "none") m = NV_ENC_BFRAME_REF_MODE_DISABLED;
			else if (v == "middle") m = NV_ENC_BFRAME_REF_MODE_MIDDLE;
			else if (v == "each") m = NV_ENC_BFRAME_REF_MODE_EACH;
			else if (v == "hierarchical") m = NV_ENC_BFRAME_REF_MODE_HIERARCHICAL;
			else { fprintf(stderr, "nvenc: bad b_reference_mode '%s' (none, middle, each or hierarchical)\n", v.c_str()); return false; }
			if (codec == NV_ENC_CODEC_H264_GUID) cfg.encodeCodecConfig.h264Config.useBFramesAsRef = m;
			else if (codec == NV_ENC_CODEC_HEVC_GUID) cfg.encodeCodecConfig.hevcConfig.useBFramesAsRef = m;
			else cfg.encodeCodecConfig.av1Config.useBFramesAsRef = m;
		}
		if (take("max_ref_frames", v)) {
			if (codec == NV_ENC_CODEC_H264_GUID) cfg.encodeCodecConfig.h264Config.maxNumRefFrames = atoi(v.c_str());
			else if (codec == NV_ENC_CODEC_HEVC_GUID) cfg.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB = atoi(v.c_str());
			else cfg.encodeCodecConfig.av1Config.maxNumRefFramesInDPB = atoi(v.c_str());
		}
		if (take("num_ltr_frames", v)) { // long-term references: 0 switches them off, above 0 switches them on with that many frames
			const uint32_t n = (uint32_t)atoi(v.c_str());
			if (codec == NV_ENC_CODEC_H264_GUID) { auto& c = cfg.encodeCodecConfig.h264Config; c.enableLTR = c.ltrTrustMode = n > 0; c.ltrNumFrames = n; }
			else if (codec == NV_ENC_CODEC_HEVC_GUID) { auto& c = cfg.encodeCodecConfig.hevcConfig; c.enableLTR = c.ltrTrustMode = n > 0; c.ltrNumFrames = n; }
			else { auto& c = cfg.encodeCodecConfig.av1Config; c.enableLTR = n > 0; c.ltrNumFrames = n; }
		}

		NV_ENC_INITIALIZE_PARAMS ip{};

		Fields fields = fields_for(codec, &cfg, &ip); // every other option
		for (auto& [k, val] : o) {
			auto it = fields.find(k);
			if (it == fields.end()) { fprintf(stderr, "nvenc: unsupported option '%s'\n", k.c_str()); return false; }
			long long n;
			std::string level = val;
			if (k == "level") { // "4.1", "auto": see options.h
				std::string why;
				if (!translate_named_value("nvenc", p.codec, "level", level, why)) { fprintf(stderr, "nvenc: %s\n", why.c_str()); return false; }
			}
			if (k == "tier" && (val == "main" || val == "high")) n = val == "high"; // the standard names, see options.h
			else if (k == "level") n = atoll(level.c_str());
			else if (val == "true" || val == "on") n = 1;
			else if (val == "false" || val == "off") n = 0;
			else {
				char* end = nullptr;
				n = strtoll(val.c_str(), &end, 0);
				if (end == val.c_str() || *end) { fprintf(stderr, "nvenc: bad value '%s' for '%s'\n", val.c_str(), k.c_str()); return false; }
			}
			it->second.set(n);
		}
		if (rc.lookaheadLevel != 0 && !rc.enableLookahead) { // NVIDIA's user-mode library aborts the process on this (fast-fail) instead of returning an error
			fprintf(stderr, "nvenc: lookaheadLevel needs lookahead (rc-lookahead above 0)\n");
			return false;
		}

		ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
		ip.encodeGUID = codec;
		ip.presetGUID = *PRESETS[preset - 1];
		ip.tuningInfo = tune;
		ip.encodeWidth = ip.darWidth = w;
		ip.encodeHeight = ip.darHeight = h;
		ip.frameRateNum = p.framerate.num;
		ip.frameRateDen = p.framerate.den;
		ip.enablePTD = 1;
		ip.enableEncodeAsync = ASYNC;
		ip.encodeConfig = &cfg;
		used_cfg = cfg; used_ip = ip; used_ip.encodeConfig = nullptr; used_codec = codec; used_preset = preset; used_tune = tune;
		if ((st = f.nvEncInitializeEncoder(enc, &ip))) return fail(st, "initialize");

		// Enough buffers to cover the reorder and lookahead delay.
		delay = cfg.frameIntervalP + rc.lookaheadDepth + 3;
		b_delay = (int)cfg.frameIntervalP > 1 ? (int)cfg.frameIntervalP - 1 : 0;
		slots.resize(delay + 16);
		for (auto& sl : slots) { // (input buffers are made when a slot first takes a frame from memory: frames from VRAM need none)
			NV_ENC_CREATE_BITSTREAM_BUFFER cb{};
			cb.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
			if ((st = f.nvEncCreateBitstreamBuffer(enc, &cb))) return fail(st, "create bitstream buffer");
			sl.out = cb.bitstreamBuffer;
			free_.push_back(&sl);
		}
#ifdef _WIN32
		auto make_event = [&](void*& ev) {
			ev = CreateEventW(nullptr, FALSE, FALSE, nullptr);
			NV_ENC_EVENT_PARAMS ep{};
			ep.version = NV_ENC_EVENT_PARAMS_VER;
			ep.completionEvent = ev;
			return ev && f.nvEncRegisterAsyncEvent(enc, &ep) == NV_ENC_SUCCESS;
		};
		for (auto& sl : slots) if (!make_event(sl.ev)) { fprintf(stderr, "nvenc: cannot register a completion event\n"); return false; }
		if (!make_event(eos_event)) { fprintf(stderr, "nvenc: cannot register a completion event\n"); return false; }
		rx = std::thread([this] { receive(); });
#endif
		return true;
	}

#ifndef VES_NO_VULKAN
	// Frees what gin_open() made in CUDA (after the encoder let go of the registrations) and then the Vulkan buffers.
	void gin_free() {
		if (!gin) return;
		if (ctx && !cu->cuCtxPushCurrent(ctx)) {
			for (auto& r : gin->res) {
				if (r.ptr) cu->cuMemFree(r.ptr);
				if (r.ext) cu->cuDestroyExternalMemory(r.ext);
			}
			CUcontext dummy;
			cu->cuCtxPopCurrent(&dummy);
		}
		gin.reset();
	}

	// For the first frame on the GPU: whether frames can come straight from VRAM (the frame's Vulkan device is the encoder's GPU, Vulkan can export
	// memory, CUDA can import it, NVENC takes it), and the buffers for it. False (and through memory from then on, said once) when not.
	bool gin_open(const VkDecodedFrame& bf) {
		gin_state = -1;
		auto no = [&](const std::string& why) {
			fprintf(stderr, "nvenc: frames come through memory: %s\n", why.c_str());
			gin_free();
			return false;
		};
		if (const char* e = getenv("VES_ENC_GPU_INPUT"); e && atoi(e) == 0) return no("VES_ENC_GPU_INPUT=0");
		if (!cu->cuImportExternalMemory || !cu->cuExternalMemoryGetMappedBuffer || !cu->cuDestroyExternalMemory || !cu->cuDeviceGetUuid)
			return no("the CUDA driver cannot import memory");
		std::string why;
		const auto devs = Vk::devices(why);
		if (bf.device < 0 || bf.device >= (int)devs.size() || !devs[(size_t)bf.device].uuid_valid) return no("the frames' Vulkan device is unknown");
		CUuuid u;
		if (cu->cuDeviceGetUuid(&u, cudev)) return no("CUDA has no UUID for the GPU");
		if (memcmp(&u, devs[(size_t)bf.device].uuid, 16)) return no("the frames are on another GPU (" + devs[(size_t)bf.device].name + ")");
		auto g = std::make_unique<GpuInput>();
		try {
			g->vk = std::make_unique<Vk>(bf.device, 1);
		} catch (const std::exception& e) { return no(e.what()); }
		g->device = bf.device;
		const int bps = depth > 8 ? 2 : 1;
		g->pitch = (uint32_t)(((size_t)w * bps + 255) & ~(size_t)255);
		g->plane_off[0] = 0;
		g->plane_off[1] = (VkDeviceSize)g->pitch * h;
		g->plane_off[2] = (VkDeviceSize)g->pitch * h * 2;
		g->size = chroma == 444 ? (VkDeviceSize)g->pitch * h * 3 : (VkDeviceSize)g->pitch * h + (VkDeviceSize)g->pitch * ((h + 1) / 2);
		gin = std::move(g);
		gin->res.resize(slots.size());
		if (cu->cuCtxPushCurrent(ctx)) return no("CUDA context");
		bool ok = true;
		for (auto& r : gin->res) {
			if (!gin->vk->exportable_buffer(gin->size, r.eb, why)) { ok = false; break; }
			CUDA_EXTERNAL_MEMORY_HANDLE_DESC md{};
#ifdef _WIN32
			md.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32;
			md.handle.win32.handle = r.eb.win32;
#else
			md.type = CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD;
			md.handle.fd = r.eb.fd;
#endif
			md.size = r.eb.alloc_size;
			md.flags = CUDA_EXTERNAL_MEMORY_DEDICATED_FLAG;
			const CUresult cr = cu->cuImportExternalMemory(&r.ext, &md);
#ifdef _WIN32
			CloseHandle((HANDLE)r.eb.win32); // CUDA holds its own reference
#else
			if (cr) close(r.eb.fd); // a successful import owns the fd
#endif
			if (cr) { r.ext = nullptr; why = "cuImportExternalMemory failed (" + std::to_string((int)cr) + ")"; ok = false; break; }
			CUDA_EXTERNAL_MEMORY_BUFFER_DESC bd{};
			bd.offset = 0;
			bd.size = gin->size;
			if (const CUresult c2 = cu->cuExternalMemoryGetMappedBuffer(&r.ptr, r.ext, &bd)) { r.ptr = 0; why = "cuExternalMemoryGetMappedBuffer failed (" + std::to_string((int)c2) + ")"; ok = false; break; }
			NV_ENC_REGISTER_RESOURCE rr{};
			rr.version = NV_ENC_REGISTER_RESOURCE_VER;
			rr.resourceType = NV_ENC_INPUT_RESOURCE_TYPE_CUDADEVICEPTR;
			rr.resourceToRegister = (void*)r.ptr;
			rr.width = (uint32_t)w; rr.height = (uint32_t)h;
			rr.pitch = gin->pitch;
			rr.bufferFormat = buffer_fmt;
			rr.bufferUsage = NV_ENC_INPUT_IMAGE;
			if (const NVENCSTATUS st = f.nvEncRegisterResource(enc, &rr)) { why = "nvEncRegisterResource failed (status " + std::to_string((int)st) + ")"; ok = false; break; }
			r.reg = rr.registeredResource;
		}
		CUcontext dummy;
		cu->cuCtxPopCurrent(&dummy);
		if (!ok) {
			for (auto& r : gin->res) if (r.reg) { f.nvEncUnregisterResource(enc, r.reg); r.reg = nullptr; }
			return no(why);
		}
		gin_state = 1;
		fprintf(stderr, "nvenc: frames come straight from VRAM (Vulkan device %d, packed on the GPU, read by NVENC through CUDA)\n", bf.device);
		return true;
	}

	// Packs a buffer frame into slot `k`'s registered buffer and waits until that is done. False (the run fails) when the GPU fails.
	bool gin_pack(const VkDecodedFrame& bf, size_t k) {
		GpuInput& g = *gin;
		GpuInput::Res& r = g.res[k];
		const int bytes = depth > 8 ? 2 : 1;
		const int cw = chroma == 444 ? w : (w + 1) / 2, ch = chroma == 444 ? h : (h + 1) / 2;
		if (bf.device != g.device || bf.nplanes != 3 || bf.bytes != bytes || (int)bf.planes[0].w != w || (int)bf.planes[0].h != h || (int)bf.planes[1].w != cw ||
		    (int)bf.planes[1].h != ch || (int)bf.planes[2].w != cw || (int)bf.planes[2].h != ch) {
			fprintf(stderr, "nvenc: a frame from VRAM is not the %dx%d %d-bit %d the encoder takes\n", w, h, depth, chroma);
			return false;
		}
		VkDeviceSize need = 0;
		for (int i = 0; i < 3; i++) need = std::max<VkDeviceSize>(need, bf.buf_offset[i] + (VkDeviceSize)bf.planes[i].w * bf.planes[i].h * bytes);
		need = (need + 3) & ~(VkDeviceSize)3;
		try {
			if (g.stage_size < need) { // the first frame (or one laid out larger): the pipelines bind the staging buffer, so they are made again
				for (auto& x : g.res) if (x.has_pipe) { g.vk->release(x.pipe); x.has_pipe = false; }
				g.stage = g.vk->buffer(need, 1);
				g.stage_size = need;
			}
			if (!r.has_pipe) { r.pipe = g.vk->pipeline("enc_pack", {&g.stage, &r.eb.b}); r.has_pipe = true; }
			struct { int32_t w, h, bytes, shift, mode, src_a, src_b, dst_base, pitch_words; } pc{};
			pc.bytes = bytes;
			pc.shift = depth > 8 ? 16 - depth : 0;
			pc.pitch_words = (int32_t)(g.pitch / 4);
			auto run = [&](int mode, int pw, int ph, VkDeviceSize a, VkDeviceSize b, VkDeviceSize dst) {
				pc.w = pw; pc.h = ph; pc.mode = mode;
				pc.src_a = (int32_t)a; pc.src_b = (int32_t)b; pc.dst_base = (int32_t)(dst / 4);
				const int row = mode == 1 ? pw * 2 : pw, per = 4 / bytes;
				g.vk->bind(r.pipe);
				g.vk->push(&pc, sizeof pc);
				g.vk->dispatch((uint32_t)(((row + per - 1) / per + 15) / 16), (uint32_t)((ph + 15) / 16));
			};
			g.vk->begin(0);
			g.vk->wait_timeline(0, bf.buf_sem, bf.buf_value);
			g.vk->copy_rows(bf.buffer, 0, need, g.stage.buf, 0, need, 1);
			g.vk->barrier();
			run(0, w, h, bf.buf_offset[0], 0, g.plane_off[0]);
			if (chroma == 444) {
				run(0, cw, ch, bf.buf_offset[1], 0, g.plane_off[1]);
				run(0, cw, ch, bf.buf_offset[2], 0, g.plane_off[2]);
			} else run(1, cw, ch, bf.buf_offset[1], bf.buf_offset[2], g.plane_off[1]);
			g.vk->barrier();
			g.vk->submit(0);
			g.vk->wait(0);
		} catch (const std::exception& e) {
			fprintf(stderr, "nvenc: packing a frame on the GPU: %s\n", e.what());
			return false;
		}
		return true;
	}
#endif

	// Copies the planar frame into the layout NVENC wants: NV12 / P010 (interleaved chroma) or planar 4:4:4.
	// 10-bit samples go in the top bits of 16, as NVENC expects, so each one is shifted left by 6.
	void repack(const Picture* fr, uint8_t* dst, uint32_t pitch) const {
		pack_plane(*fr, 0, w, h, depth, dst, pitch);
		if (chroma == 444) {
			pack_plane(*fr, 1, w, h, depth, dst + (size_t)pitch * h, pitch);
			pack_plane(*fr, 2, w, h, depth, dst + (size_t)pitch * h * 2, pitch);
			return;
		}
		pack_uv(*fr, w, h, depth, dst + (size_t)pitch * h, pitch);
	}

	// Reads the finished output of a slot and hands it to the sink.
	bool read_slot(Slot* s) {
		NV_ENC_LOCK_BITSTREAM lb{};
		lb.version = NV_ENC_LOCK_BITSTREAM_VER;
		lb.outputBitstream = s->out;
		NVENCSTATUS st = f.nvEncLockBitstream(enc, &lb);
		if (st) return fail(st, "lock bitstream");
		stats.packets++; stats.bytes += lb.bitstreamSizeInBytes;
		outputs++;
		if (sink) sink((const uint8_t*)lb.bitstreamBufferPtr, lb.bitstreamSizeInBytes, (int64_t)lb.outputTimeStamp,
			lb.pictureType == NV_ENC_PIC_TYPE_IDR || lb.pictureType == NV_ENC_PIC_TYPE_I, NO_DTS);
		f.nvEncUnlockBitstream(enc, s->out);
		return true;
	}

#ifdef _WIN32
	// Waits for the event, giving up when the encoder is being torn down.
	bool wait(void* ev) {
		for (;;) {
			if (WaitForSingleObject((HANDLE)ev, 20) == WAIT_OBJECT_0) return true;
			std::lock_guard lock(m);
			if (stop) return false;
		}
	}

	void receive() {
		for (;;) {
			Slot* s = nullptr;
			{
				std::unique_lock lock(m);
				cv.wait(lock, [&] { return !pending_.empty() || flushing || stop; });
				if (stop) { finished = true; rx_ok = false; cv.notify_all(); return; }
				if (!pending_.empty()) s = pending_.front();
			}
			if (!s) { // everything submitted has been read; the end-of-stream picture is what is left
				bool ok = wait(eos_event);
				std::lock_guard lock(m);
				finished = true;
				rx_ok = rx_ok && ok;
				cv.notify_all();
				return;
			}
			bool ok = wait(s->ev) && read_slot(s);
			std::lock_guard lock(m);
			if (!ok) { rx_ok = false; finished = true; cv.notify_all(); return; }
			pending_.pop_front();
			free_.push_back(s);
			cv.notify_all();
		}
	}
#else
	// Synchronous mode: outputs are read in submission order, `count` of them, on the calling thread.
	bool drain(size_t count) {
		while (count-- && !pending_.empty()) {
			Slot* s = pending_.front();
			pending_.pop_front();
			free_.push_back(s);
			if (!read_slot(s)) return false;
		}
		return true;
	}
#endif

	// Waits (with m held) until `done`, as long as output keeps coming: after NO_OUTPUT without a single picture read it gives up, stops the receiver and
	// fails the encoder (false).
	template <class Pred> bool wait_output(std::unique_lock<std::mutex>& lock, Pred done, const char* what) {
		auto since = std::chrono::steady_clock::now();
		int64_t seen = outputs.load();
		while (!done()) {
			cv.wait_for(lock, std::chrono::milliseconds(500));
			const auto now = std::chrono::steady_clock::now();
			if (const int64_t o = outputs.load(); o != seen) { seen = o; since = now; continue; }
			if (now - since >= NO_OUTPUT && !done()) {
				fprintf(stderr, "nvenc: no output for %lld s while waiting for %s (%lld pictures given, %lld finished): the encoder takes frames but gives nothing back with these settings\n",
					(long long)std::chrono::duration_cast<std::chrono::seconds>(NO_OUTPUT).count(), what, submitted, (long long)seen);
				stop = true; // the receiver gives up waiting for the event (wait() checks it)
				rx_ok = false;
				cv.notify_all();
				return false;
			}
		}
		return true;
	}

	int reorder_delay() const override { return b_delay; }

	// The options open() reads itself, then the fields, from what it handed the encoder.
	std::vector<std::pair<std::string, std::string>> settings() override {
		if (!enc) return {};
		std::vector<std::pair<std::string, std::string>> out;
		auto add = [&](const std::string& k, const std::string& v) { out.push_back({k, v}); };
		NV_ENC_CONFIG cfg = used_cfg;
		NV_ENC_INITIALIZE_PARAMS ip = used_ip;
		const NV_ENC_RC_PARAMS& rc = cfg.rcParams;
		const bool h264 = used_codec == NV_ENC_CODEC_H264_GUID, hevc = used_codec == NV_ENC_CODEC_HEVC_GUID;
		add("preset", "p" + std::to_string(used_preset));
		for (auto& [name, t] : tunes()) if (t == used_tune) add("tune", name);
		std::string profile = "auto";
		for (auto& [name, guid] : profiles(h264 ? "h264" : hevc ? "hevc" : "av1")) if (*guid == cfg.profileGUID) profile = name;
		add("profile", profile);
		add("rate_control", rc.rateControlMode == NV_ENC_PARAMS_RC_CBR ? "cbr" : rc.rateControlMode == NV_ENC_PARAMS_RC_VBR ? "vbr" : std::to_string((int)rc.rateControlMode));
		add("max_b_frames", std::to_string(cfg.frameIntervalP > 0 ? (int)cfg.frameIntervalP - 1 : 0));
		add("lookahead_frames", std::to_string(rc.enableLookahead ? (int)rc.lookaheadDepth : 0));
		add("spatial_aq", !rc.enableAQ ? "0" : rc.aqStrength == 0 ? "auto" : std::to_string((int)rc.aqStrength));
		add("temporal_aq", std::to_string((int)rc.enableTemporalAQ));
		{
			const int m = h264 ? (int)cfg.encodeCodecConfig.h264Config.useBFramesAsRef : hevc ? (int)cfg.encodeCodecConfig.hevcConfig.useBFramesAsRef
			                   : (int)cfg.encodeCodecConfig.av1Config.useBFramesAsRef;
			add("b_reference_mode", m == NV_ENC_BFRAME_REF_MODE_EACH ? "each" : m == NV_ENC_BFRAME_REF_MODE_MIDDLE ? "middle" : m == NV_ENC_BFRAME_REF_MODE_HIERARCHICAL ? "hierarchical"
			                        : m == NV_ENC_BFRAME_REF_MODE_DISABLED ? "none" : std::to_string(m));
		}
		add("max_ref_frames", std::to_string(h264 ? (int)cfg.encodeCodecConfig.h264Config.maxNumRefFrames : hevc ? (int)cfg.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB
		                                           : (int)cfg.encodeCodecConfig.av1Config.maxNumRefFramesInDPB));
		add("num_ltr_frames", std::to_string(h264 ? (int)cfg.encodeCodecConfig.h264Config.ltrNumFrames : hevc ? (int)cfg.encodeCodecConfig.hevcConfig.ltrNumFrames
		                                           : (int)cfg.encodeCodecConfig.av1Config.ltrNumFrames));
		for (auto& [name, f] : fields_for(used_codec, &cfg, &ip)) {
			// these also switch their limit on: off is not the same as any value
			const bool off = (name == "min_qp" && !rc.enableMinQP) || (name == "max_qp" && !rc.enableMaxQP) || (name == "initial_rc_qp" && !rc.enableInitialRCQP);
			add(name, off ? "off" : std::to_string(f.get()));
		}
		return out;
	}

	bool encode(const Picture* fr) override {
		NVENCSTATUS st;
		if (!fr) { // flush
			{ std::lock_guard lock(m); if (!rx_ok || stop) return false; } // gave up already: nothing more will come
			NV_ENC_PIC_PARAMS pp{};
			pp.version = NV_ENC_PIC_PARAMS_VER;
			pp.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
			pp.completionEvent = eos_event;
			if ((st = f.nvEncEncodePicture(enc, &pp))) return fail(st, "flush");
#ifndef _WIN32
			return drain(pending_.size());
#else
			std::unique_lock lock(m);
			flushing = true;
			cv.notify_all();
			if (!wait_output(lock, [&] { return finished; }, "the end of the stream")) return false;
			return rx_ok;
#endif
		}
		Slot* s;
		{
			std::unique_lock lock(m);
#ifndef _WIN32
			if (free_.empty()) { // synchronous: only this thread reads outputs, so none will come
				fprintf(stderr, "nvenc: no output after %lld pictures: the encoder takes frames but gives nothing back with these settings\n", submitted);
				return false;
			}
#endif
			// never empties for good: the slots outnumber what the encoder holds back
			if (!wait_output(lock, [&] { return !free_.empty() || !rx_ok; }, "a free picture buffer")) return false;
			if (!rx_ok) return false;
			s = free_.back(); free_.pop_back();
		}

		if (s->mapped) { f.nvEncUnmapInputResource(enc, s->mapped); s->mapped = nullptr; } // its last picture is done (its output was read)
		NV_ENC_INPUT_PTR input = nullptr;
		uint32_t pitch = 0;
		Picture mem; // a frame on the GPU read back, when it cannot be taken from VRAM
#ifndef VES_NO_VULKAN
		if (fr->gpu && gin_state >= 0) {
			const VkDecodedFrame* bf = (const VkDecodedFrame*)fr->gpu.get();
			if (bf->buffer && (gin_state == 1 || gin_open(*bf))) {
				if (!gin_pack(*bf, (size_t)(s - slots.data()))) return false;
				NV_ENC_MAP_INPUT_RESOURCE mi{};
				mi.version = NV_ENC_MAP_INPUT_RESOURCE_VER;
				mi.registeredResource = gin->res[(size_t)(s - slots.data())].reg;
				if ((st = f.nvEncMapInputResource(enc, &mi))) return fail(st, "map input resource");
				s->mapped = input = mi.mappedResource;
				pitch = gin->pitch;
			}
		}
#endif
		if (!input) { // from memory (a frame on the GPU is read back first)
			if (fr->gpu) {
				mem = *fr;
				if (!picture_to_memory(mem)) return false;
				fr = &mem;
			}
			if (!s->in) {
				NV_ENC_CREATE_INPUT_BUFFER ci{};
				ci.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
				ci.width = w; ci.height = h;
				ci.bufferFmt = buffer_fmt;
				if ((st = f.nvEncCreateInputBuffer(enc, &ci))) return fail(st, "create input buffer");
				s->in = ci.inputBuffer;
			}
			NV_ENC_LOCK_INPUT_BUFFER li{};
			li.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
			li.inputBuffer = s->in;
			if ((st = f.nvEncLockInputBuffer(enc, &li))) return fail(st, "lock input");
			repack(fr, (uint8_t*)li.bufferDataPtr, li.pitch);
			f.nvEncUnlockInputBuffer(enc, s->in);
			input = s->in;
			pitch = li.pitch;
		}

		NV_ENC_PIC_PARAMS pp{};
		pp.version = NV_ENC_PIC_PARAMS_VER;
		pp.inputBuffer = input;
		pp.bufferFmt = buffer_fmt;
		pp.inputWidth = w; pp.inputHeight = h; pp.inputPitch = pitch;
		pp.outputBitstream = s->out;
		if (ASYNC) pp.completionEvent = s->ev;
		pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
		pp.inputTimeStamp = fr->pts;
		{ // queue it before submitting: the event can fire as soon as the call is made
			std::lock_guard lock(m);
			pending_.push_back(s);
		}
		cv.notify_all();
		submitted++;
		st = f.nvEncEncodePicture(enc, &pp);
#ifndef _WIN32
		if (st == NV_ENC_SUCCESS) return drain(1); // each success makes the oldest queued picture's output available
#endif
		if (st == NV_ENC_SUCCESS || st == NV_ENC_ERR_NEED_MORE_INPUT) return true;
		return fail(st, "encode");
	}
};

} // namespace

std::unique_ptr<Encoder> make_encoder_nvenc() { return std::make_unique<Nvenc>(); }

int nvenc_cuda_ordinal(int gpu) {
	CudaFunctions* cu = nullptr;
	int ordinal = -1;
	CUdevice dev;
	if (cuda_load_functions(&cu, nullptr) >= 0 && !cu->cuInit(0) && cuda_device_for(cu, gpu, dev)) ordinal = (int)dev; // a CUdevice is CUDA's ordinal
	cuda_free_functions(&cu);
	return ordinal;
}

// All capabilities of the NVENC on the n-th NVIDIA GPU for a codec, as a JSON object; "null" if it cannot be asked.
std::string nvenc_caps_json(const std::string& codec, int gpu) {
	bool known;
	GUID guid = codec_guid(codec, known);
	CudaFunctions* cu = nullptr;
	NvencFunctions* nv = nullptr;
	CUcontext ctx = nullptr;
	void* enc = nullptr;
	NV_ENCODE_API_FUNCTION_LIST f{};
	CUdevice dev;
	std::string out = "null";
	if (known && cuda_load_functions(&cu, nullptr) >= 0 && nvenc_load_functions(&nv, nullptr) >= 0 && !cu->cuInit(0) && cuda_device_for(cu, gpu, dev) &&
	    !cu->cuCtxCreate(&ctx, 0, dev)) {
		f.version = NV_ENCODE_API_FUNCTION_LIST_VER;
		NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS os{};
		os.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
		os.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
		os.device = ctx;
		os.apiVersion = NVENCAPI_VERSION;
		if (!nv->NvEncodeAPICreateInstance(&f) && !f.nvEncOpenEncodeSessionEx(&os, &enc)) {
			out = "{";
			bool first = true;
			auto add = [&](const char* name, NV_ENC_CAPS c) {
				NV_ENC_CAPS_PARAM cp{};
				cp.version = NV_ENC_CAPS_PARAM_VER;
				cp.capsToQuery = c;
				int v = 0;
				if (f.nvEncGetEncodeCaps(enc, guid, &cp, &v)) return; // not available for this codec or driver: left out
				out += std::string(first ? "\"" : ",\"") + name + "\":" + std::to_string(v);
				first = false;
			};
#define X(n) add(#n, NV_ENC_CAPS_##n);
			NVENC_CAPS(X)
#undef X
			out += "}";
			f.nvEncDestroyEncoder(enc);
		}
	}
	if (ctx) cu->cuCtxDestroy(ctx);
	nvenc_free_functions(&nv);
	cuda_free_functions(&cu);
	return out;
}


// The options open() reads itself (the `take` calls), then the fields.
OptionList nvenc_options(const std::string& codec) {
	OptionList out = {
		{"preset", "p1 .. p7", "p1 fastest, p7 slowest (default p4)"},
		{"tune", "", "default hq"},
		{"profile", "", "left out, the encoder picks one that fits the chroma layout and bit depth"},
		{"rate_control", "cbr | vbr", "default cbr"},
		{"max_b_frames", "0..", "most B-frames in a row (0 = none)"},
		{"lookahead_frames", "0..", "frames of lookahead for rate control (0 = off)"},
		{"spatial_aq", "0 | auto | 1..15", "0 off, auto lets the driver pick the strength per frame"},
		{"temporal_aq", "0 | 1", ""},
	};
	for (auto& [name, t] : tunes()) out[1].values += (out[1].values.empty() ? "" : " | ") + name;
	for (auto& [name, guid] : profiles(codec)) out[2].values += (out[2].values.empty() ? "" : " | ") + name;
	out.push_back({"b_reference_mode", "none | middle | each | hierarchical", "B-frames as references"});
	out.push_back({"max_ref_frames", "1..", "most reference frames"});
	out.push_back({"num_ltr_frames", "0..", "long-term references: 0 off, above 0 on with that many frames"});
	bool known;
	const GUID guid = codec_guid(codec, known);
	if (!known) return {};
	NV_ENC_CONFIG cfg{};
	NV_ENC_INITIALIZE_PARAMS ip{};
	const Fields fields = fields_for(guid, &cfg, &ip);
	for (auto& [name, f] : fields) {
		auto v = field_values().find(name);
		out.push_back({name, v != field_values().end() ? v->second : f.is_enum ? "number (an enum of nvEncodeAPI.h)" : field_range(f), f.native});
	}
	return out;
}

bool nvenc_supports(const std::string& codec, int chroma, int depth) {
	static std::mutex m;
	static std::map<std::string, Caps> cache;
	Caps c;
	{
		std::lock_guard<std::mutex> lock(m);
		auto it = cache.find(codec);
		if (it == cache.end()) it = cache.emplace(codec, probe_caps(codec)).first;
		c = it->second;
	}
	if (chroma == 422 || depth > 10) return false;
	if (depth == 10 && !c.b10) return false;
	return chroma == 420 || c.c444;
}
