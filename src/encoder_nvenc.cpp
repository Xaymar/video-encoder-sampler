// Native NVENC backend: talks to nvEncodeAPI directly (loaded at runtime), no libavcodec.
#include "encoder.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <vector>

extern "C" {
#include <ffnvcodec/dynlink_loader.h>
#include <libswscale/swscale.h>
}

namespace {

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
	SwsContext* sws = nullptr;
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
		sws_freeContext(sws);
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

		GUID codec;
		bool h26x = true;
		if (p.codec == "h264") codec = NV_ENC_CODEC_H264_GUID;
		else if (p.codec == "hevc") codec = NV_ENC_CODEC_HEVC_GUID;
		else if (p.codec == "av1") { codec = NV_ENC_CODEC_AV1_GUID; h26x = false; }
		else { fprintf(stderr, "nvenc: unknown codec '%s'\n", p.codec.c_str()); return false; }

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
			ci.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
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
		if (!sws) sws = sws_getContext(fr->width, fr->height, (AVPixelFormat)fr->format, w, h, AV_PIX_FMT_NV12, SWS_BICUBIC, nullptr, nullptr, nullptr);
		uint8_t* dst[4] = {(uint8_t*)li.bufferDataPtr, (uint8_t*)li.bufferDataPtr + (size_t)li.pitch * h, nullptr, nullptr};
		int stride[4] = {(int)li.pitch, (int)li.pitch, 0, 0};
		sws_scale(sws, fr->data, fr->linesize, 0, fr->height, dst, stride);
		f.nvEncUnlockInputBuffer(enc, s->in);

		NV_ENC_PIC_PARAMS pp{};
		pp.version = NV_ENC_PIC_PARAMS_VER;
		pp.inputBuffer = s->in;
		pp.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
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
