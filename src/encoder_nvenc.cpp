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
#include <vector>

#include "formats.h"
#include "gpu.h"
#include "repack.h"

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
	NV_ENC_INPUT_PTR in = nullptr;
	NV_ENC_OUTPUT_PTR out = nullptr;
	void* ev = nullptr; // signaled when the encoder is done with this slot's picture
};

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
	NV_ENC_BUFFER_FORMAT buffer_fmt = NV_ENC_BUFFER_FORMAT_NV12;
	int chroma = 420, depth = 8;
	Sink sink;
	int w = 0, h = 0;
	size_t delay = 0;
	int b_delay = 0;

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
				if (s.in) f.nvEncDestroyInputBuffer(enc, s.in);
				if (s.out) f.nvEncDestroyBitstreamBuffer(enc, s.out);
			}
			f.nvEncDestroyEncoder(enc);
		}
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
		CUdevice dev;
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
		if ((st = f.nvEncOpenEncodeSessionEx(&os, &enc))) { enc = nullptr; return fail(st, "open session"); }

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
			if (v == "hq") tune = NV_ENC_TUNING_INFO_HIGH_QUALITY;
			else if (v == "ll") tune = NV_ENC_TUNING_INFO_LOW_LATENCY;
			else if (v == "ull") tune = NV_ENC_TUNING_INFO_ULTRA_LOW_LATENCY;
			else if (v == "lossless") tune = NV_ENC_TUNING_INFO_LOSSLESS;
			else { fprintf(stderr, "nvenc: bad tune '%s'\n", v.c_str()); return false; }
		}

		NV_ENC_PRESET_CONFIG pc{};
		pc.version = NV_ENC_PRESET_CONFIG_VER;
		pc.presetCfg.version = NV_ENC_CONFIG_VER;
		if ((st = f.nvEncGetEncodePresetConfigEx(enc, codec, *PRESETS[preset - 1], tune, &pc))) return fail(st, "get preset config");
		NV_ENC_CONFIG cfg = pc.presetCfg;

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

		if (take("rc", v)) {
			if (v == "cbr") rc.rateControlMode = NV_ENC_PARAMS_RC_CBR;
			else if (v == "vbr") rc.rateControlMode = NV_ENC_PARAMS_RC_VBR;
			else { fprintf(stderr, "nvenc: bad rc '%s'\n", v.c_str()); return false; }
		}
		if (take("bf", v)) cfg.frameIntervalP = atoi(v.c_str()) + 1;
		if (take("rc-lookahead", v)) { rc.lookaheadDepth = (uint16_t)atoi(v.c_str()); rc.enableLookahead = rc.lookaheadDepth > 0; }
		if (take("spatial-aq", v)) rc.enableAQ = atoi(v.c_str()) != 0;
		if (take("temporal-aq", v)) rc.enableTemporalAQ = atoi(v.c_str()) != 0;
		if (take("aq-strength", v)) rc.aqStrength = atoi(v.c_str());
		if (take("b_ref_mode", v)) {
			NV_ENC_BFRAME_REF_MODE m = v == "each" ? NV_ENC_BFRAME_REF_MODE_EACH : v == "middle" ? NV_ENC_BFRAME_REF_MODE_MIDDLE : NV_ENC_BFRAME_REF_MODE_DISABLED;
			if (codec == NV_ENC_CODEC_H264_GUID) cfg.encodeCodecConfig.h264Config.useBFramesAsRef = m;
			else if (codec == NV_ENC_CODEC_HEVC_GUID) cfg.encodeCodecConfig.hevcConfig.useBFramesAsRef = m;
			else { fprintf(stderr, "nvenc: b_ref_mode not supported for %s\n", p.codec.c_str()); return false; }
		}
		NV_ENC_INITIALIZE_PARAMS ip{};

		// Every other option is a field of NVENC's own structures, under the field's name (see the SDK's nvEncodeAPI.h), set to a number (or
		// true/false); enums are their numeric values. Fields of the codec being used only. A QP option sets all three QPs of NV_ENC_QP.
		std::map<std::string, std::function<void(long long)>> fields;
		NV_ENC_RC_PARAMS& rcp = cfg.rcParams;
#define RC(f) fields[#f] = [&](long long v) { rcp.f = (decltype(rcp.f))v; };
		RC(maxBitRate) RC(vbvBufferSize) RC(vbvInitialDelay) RC(enableMinQP) RC(enableMaxQP) RC(enableInitialRCQP) RC(enableAQ) RC(enableLookahead)
		RC(disableIadapt) RC(disableBadapt) RC(enableTemporalAQ) RC(zeroReorderDelay) RC(enableNonRefP) RC(strictGOPTarget) RC(aqStrength)
		RC(targetQuality) RC(targetQualityLSB) RC(lookaheadDepth) RC(lowDelayKeyFrameScale) RC(yDcQPIndexOffset) RC(uDcQPIndexOffset)
		RC(vDcQPIndexOffset) RC(qpMapMode) RC(multiPass) RC(cbQPIndexOffset) RC(crQPIndexOffset) RC(lookaheadLevel)
#undef RC
		auto qp = [](NV_ENC_QP& q, long long v) { q.qpInterP = q.qpInterB = q.qpIntra = (uint32_t)v; };
		fields["constQP"] = [&](long long v) { qp(rcp.constQP, v); };
		fields["minQP"] = [&](long long v) { qp(rcp.minQP, v); rcp.enableMinQP = 1; };
		fields["maxQP"] = [&](long long v) { qp(rcp.maxQP, v); rcp.enableMaxQP = 1; };
		fields["initialRCQP"] = [&](long long v) { qp(rcp.initialRCQP, v); rcp.enableInitialRCQP = 1; };
#define CF(f) fields[#f] = [&](long long v) { cfg.f = (decltype(cfg.f))v; };
		CF(monoChromeEncoding) CF(frameFieldMode) CF(mvPrecision)
#undef CF
#define IP(f) fields[#f] = [&](long long v) { ip.f = (decltype(ip.f))v; };
		IP(enableWeightedPrediction) IP(splitEncodeMode) IP(enableUniDirectionalB)
#undef IP
		if (codec == NV_ENC_CODEC_H264_GUID) {
			auto& c = cfg.encodeCodecConfig.h264Config;
#define C(f) fields[#f] = [&](long long v) { c.f = (decltype(c.f))v; };
			C(hierarchicalPFrames) C(hierarchicalBFrames) C(outputBufferingPeriodSEI) C(outputPictureTimingSEI) C(outputAUD) C(disableSPSPPS)
			C(outputRecoveryPointSEI) C(enableIntraRefresh) C(enableConstrainedEncoding) C(repeatSPSPPS) C(enableVFR) C(enableLTR)
			C(useConstrainedIntraPred) C(enableFillerDataInsertion) C(singleSliceIntraRefresh) C(qpPrimeYZeroTransformBypassFlag) C(level)
			C(disableDeblockingFilterIDC) C(adaptiveTransformMode) C(bdirectMode) C(entropyCodingMode) C(intraRefreshPeriod) C(intraRefreshCnt)
			C(maxNumRefFrames) C(sliceMode) C(sliceModeData) C(ltrNumFrames) C(ltrTrustMode) C(maxTemporalLayers) C(numTemporalLayers)
			C(enableTemporalSVC) C(numRefL0) C(numRefL1) C(tfLevel)
#undef C
		} else if (codec == NV_ENC_CODEC_HEVC_GUID) {
			auto& c = cfg.encodeCodecConfig.hevcConfig;
#define C(f) fields[#f] = [&](long long v) { c.f = (decltype(c.f))v; };
			C(level) C(tier) C(minCUSize) C(maxCUSize) C(useConstrainedIntraPred) C(disableDeblockAcrossSliceBoundary) C(outputBufferingPeriodSEI)
			C(outputPictureTimingSEI) C(outputAUD) C(enableLTR) C(disableSPSPPS) C(repeatSPSPPS) C(enableIntraRefresh) C(enableFillerDataInsertion)
			C(enableConstrainedEncoding) C(singleSliceIntraRefresh) C(outputRecoveryPointSEI) C(enableTemporalSVC) C(intraRefreshPeriod)
			C(intraRefreshCnt) C(maxNumRefFramesInDPB) C(ltrNumFrames) C(sliceMode) C(sliceModeData) C(maxTemporalLayersMinus1) C(ltrTrustMode)
			C(numRefL0) C(numRefL1) C(tfLevel) C(disableDeblockingFilterIDC) C(numTemporalLayers)
#undef C
		} else {
			auto& c = cfg.encodeCodecConfig.av1Config;
#define C(f) fields[#f] = [&](long long v) { c.f = (decltype(c.f))v; };
			C(level) C(tier) C(minPartSize) C(maxPartSize) C(outputAnnexBFormat) C(enableTimingInfo) C(enableDecoderModelInfo) C(enableFrameIdNumbers)
			C(disableSeqHdr) C(repeatSeqHdr) C(enableIntraRefresh) C(enableBitstreamPadding) C(enableCustomTileConfig) C(enableFilmGrainParams)
			C(enableLTR) C(enableTemporalSVC) C(intraRefreshPeriod) C(intraRefreshCnt) C(maxNumRefFramesInDPB) C(numTileColumns) C(numTileRows)
			C(maxTemporalLayersMinus1) C(colorPrimaries) C(transferCharacteristics) C(matrixCoefficients) C(colorRange) C(chromaSamplePosition)
			C(numFwdRefs) C(numBwdRefs) C(ltrNumFrames) C(numTemporalLayers) C(tfLevel)
#undef C
		}
		for (auto& [k, val] : o) {
			auto it = fields.find(k);
			if (it == fields.end()) { fprintf(stderr, "nvenc: unsupported option '%s'\n", k.c_str()); return false; }
			long long n;
			if (val == "true" || val == "on") n = 1;
			else if (val == "false" || val == "off") n = 0;
			else {
				char* end = nullptr;
				n = strtoll(val.c_str(), &end, 0);
				if (end == val.c_str() || *end) { fprintf(stderr, "nvenc: bad value '%s' for '%s'\n", val.c_str(), k.c_str()); return false; }
			}
			it->second(n);
		}
		if (rcp.lookaheadLevel != 0 && !rcp.enableLookahead) { // NVIDIA's user-mode library aborts the process on this (fast-fail) instead of returning an error
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
		if ((st = f.nvEncInitializeEncoder(enc, &ip))) return fail(st, "initialize");

		// Enough buffers to cover the reorder and lookahead delay.
		delay = cfg.frameIntervalP + rc.lookaheadDepth + 3;
		b_delay = (int)cfg.frameIntervalP > 1 ? (int)cfg.frameIntervalP - 1 : 0;
		slots.resize(delay + 16);
		for (auto& sl : slots) {
			NV_ENC_CREATE_INPUT_BUFFER ci{};
			ci.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
			ci.width = w; ci.height = h;
			ci.bufferFmt = buffer_fmt;
			if ((st = f.nvEncCreateInputBuffer(enc, &ci))) return fail(st, "create input buffer");
			sl.in = ci.inputBuffer;
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
				if (stop) return;
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

	int reorder_delay() const override { return b_delay; }

	bool encode(const Picture* fr) override {
		NVENCSTATUS st;
		if (!fr) { // flush
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
			cv.wait(lock, [&] { return finished; });
			return rx_ok;
#endif
		}
		Slot* s;
		{
			std::unique_lock lock(m);
			cv.wait(lock, [&] { return !free_.empty() || !rx_ok; }); // never empties for good: the slots outnumber what the encoder holds back
			if (!rx_ok) return false;
			s = free_.back(); free_.pop_back();
		}

		NV_ENC_LOCK_INPUT_BUFFER li{};
		li.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
		li.inputBuffer = s->in;
		if ((st = f.nvEncLockInputBuffer(enc, &li))) return fail(st, "lock input");
		repack(fr, (uint8_t*)li.bufferDataPtr, li.pitch);
		f.nvEncUnlockInputBuffer(enc, s->in);

		NV_ENC_PIC_PARAMS pp{};
		pp.version = NV_ENC_PIC_PARAMS_VER;
		pp.inputBuffer = s->in;
		pp.bufferFmt = buffer_fmt;
		pp.inputWidth = w; pp.inputHeight = h; pp.inputPitch = li.pitch;
		pp.outputBitstream = s->out;
		if (ASYNC) pp.completionEvent = s->ev;
		pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
		pp.inputTimeStamp = fr->pts;
		{ // queue it before submitting: the event can fire as soon as the call is made
			std::lock_guard lock(m);
			pending_.push_back(s);
		}
		cv.notify_all();
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
