// Native NVENC backend: talks to nvEncodeAPI directly (loaded at runtime), no libavcodec.
#include "encoder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <mutex>
#include <vector>

#include "formats.h"

extern "C" {
#include <ffnvcodec/dynlink_loader.h>
}

namespace {

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
	std::vector<Slot*> free_;
	std::deque<Slot*> pending; // submitted, output not yet read (encoder reorders / looks ahead)
	NV_ENC_BUFFER_FORMAT buffer_fmt = NV_ENC_BUFFER_FORMAT_NV12;
	int chroma = 420, depth = 8;
	Sink sink;
	int w = 0, h = 0;
	size_t delay = 0;

	bool fail(NVENCSTATUS s, const char* what) {
		const char* msg = enc ? f.nvEncGetLastErrorString(enc) : "";
		fprintf(stderr, "nvenc: %s failed (status %d) %s\n", what, (int)s, msg ? msg : "");
		return false;
	}

public:
	~Nvenc() override {
		if (enc) {
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
		if (cu->cuInit(0) || cu->cuDeviceGet(&dev, gpu) || cu->cuCtxCreate(&ctx, 0, dev)) {
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

		Fmt fmt;
		describe_format(p.pix, fmt);
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
		if (!o.empty()) {
			fprintf(stderr, "nvenc: unsupported option '%s'\n", o.begin()->first.c_str());
			return false;
		}

		NV_ENC_INITIALIZE_PARAMS ip{};
		ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
		ip.encodeGUID = codec;
		ip.presetGUID = *PRESETS[preset - 1];
		ip.tuningInfo = tune;
		ip.encodeWidth = ip.darWidth = w;
		ip.encodeHeight = ip.darHeight = h;
		ip.frameRateNum = p.framerate.num;
		ip.frameRateDen = p.framerate.den;
		ip.enablePTD = 1;
		ip.encodeConfig = &cfg;
		if ((st = f.nvEncInitializeEncoder(enc, &ip))) return fail(st, "initialize");

		// Enough buffers to cover the reorder and lookahead delay.
		delay = cfg.frameIntervalP + rc.lookaheadDepth + 3;
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
		return true;
	}

	// Copies the planar frame into the layout NVENC wants: NV12 / P010 (interleaved chroma) or planar 4:4:4.
	// 10-bit samples go in the top bits of 16, as NVENC expects, so each one is shifted left by 6.
	void repack(const AVFrame* fr, uint8_t* dst, uint32_t pitch) const {
		const int sample = depth > 8 ? 2 : 1;
		auto put = [&](uint8_t* d, int x, int v) { if (sample == 1) d[x] = (uint8_t)v; else ((uint16_t*)d)[x] = (uint16_t)(v << 6); };
		auto get = [&](const uint8_t* s, int x) { return sample == 1 ? (int)s[x] : (int)((const uint16_t*)s)[x]; };
		auto plane = [&](int p, uint8_t* out, int pw, int ph) {
			for (int y = 0; y < ph; y++) for (int x = 0; x < pw; x++) put(out + (size_t)y * pitch, x, get(fr->data[p] + (size_t)y * fr->linesize[p], x));
		};
		plane(0, dst, w, h);
		if (chroma == 444) {
			plane(1, dst + (size_t)pitch * h, w, h);
			plane(2, dst + (size_t)pitch * h * 2, w, h);
			return;
		}
		uint8_t* uv = dst + (size_t)pitch * h;
		for (int y = 0; y < (h + 1) / 2; y++) {
			const uint8_t *u = fr->data[1] + (size_t)y * fr->linesize[1], *v = fr->data[2] + (size_t)y * fr->linesize[2];
			uint8_t* row = uv + (size_t)y * pitch;
			for (int x = 0; x < (w + 1) / 2; x++) { put(row, 2 * x, get(u, x)); put(row, 2 * x + 1, get(v, x)); }
		}
	}

	// Outputs only become readable once the encoder has run `delay` frames past them; flush reads everything.
	bool drain(bool all = false) {
		while (!pending.empty() && (all || pending.size() > delay)) {
			Slot* s = pending.front();
			NV_ENC_LOCK_BITSTREAM lb{};
			lb.version = NV_ENC_LOCK_BITSTREAM_VER;
			lb.outputBitstream = s->out;
			NVENCSTATUS st = f.nvEncLockBitstream(enc, &lb);
			if (st) return fail(st, "lock bitstream");
			stats.packets++; stats.bytes += lb.bitstreamSizeInBytes;
			if (sink) sink((const uint8_t*)lb.bitstreamBufferPtr, lb.bitstreamSizeInBytes, (int64_t)lb.outputTimeStamp,
				lb.pictureType == NV_ENC_PIC_TYPE_IDR || lb.pictureType == NV_ENC_PIC_TYPE_I);
			f.nvEncUnlockBitstream(enc, s->out);
			pending.pop_front();
			free_.push_back(s);
		}
		return true;
	}

	bool encode(const AVFrame* fr) override {
		NVENCSTATUS st;
		if (!fr) { // flush
			NV_ENC_PIC_PARAMS pp{};
			pp.version = NV_ENC_PIC_PARAMS_VER;
			pp.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
			if ((st = f.nvEncEncodePicture(enc, &pp))) return fail(st, "flush");
			return drain(true);
		}
		if (free_.empty()) { fprintf(stderr, "nvenc: out of buffers\n"); return false; }
		Slot* s = free_.back(); free_.pop_back();

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
		pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
		pp.inputTimeStamp = fr->pts;
		st = f.nvEncEncodePicture(enc, &pp);
		pending.push_back(s);
		if (st == NV_ENC_SUCCESS || st == NV_ENC_ERR_NEED_MORE_INPUT) return drain();
		return fail(st, "encode");
	}
};

} // namespace

std::unique_ptr<Encoder> make_encoder_nvenc() { return std::make_unique<Nvenc>(); }

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
