// Reproduces an H.264 stream from NVENC whose SPS allows fewer reference frames than its slices use:
// with hierarchical P-frames on (NV_ENC_CONFIG_H264::hierarchicalPFrames = 1), NVENC writes max_num_ref_frames = 1 in the SPS
// (nvEncGetSequenceParams and the stream) whatever maxNumRefFrames is. With long-term references on as well (enableLTR = 1,
// ltrTrustMode = 1, ltrNumFrames = 4), the slices keep a short-term and a long-term reference at once, and FFmpeg's software decoder says
// "number of reference frames (1+1) exceeds max (1; probably corrupt input)". Without hierarchicalPFrames, the SPS has maxNumRefFrames.
// Measured on Irkela (RTX 4070 Laptop GPU, driver 621.24): 2448 of 2448 init-only combinations with hierarchicalPFrames = 1 gave 1, none without it.
// Results (gop 60, 30 fps): --ves --hierp on a real clip: SPS 1, 297 of 300 frames "exceeds max"; --ves: SPS 16, clean;
// --ves --hierp --ltr 0 --trust 0: SPS 1.
//
// Uses only the public NVENC API through FFmpeg's ffnvcodec headers (nv-codec-headers), which load nvcuda.dll and nvEncodeAPI64.dll at run time.
//
// Build (Windows, clang; nv-codec-headers' include folder is the only dependency; NOMINMAX because dynlink_loader.h includes windows.h):
//   clang++ -std=c++17 -O2 -DNOMINMAX -I <nv-codec-headers>/include nvenc_h264_ltr_sps.cpp -o nvenc_h264_ltr_sps.exe
// Run:
//   nvenc_h264_ltr_sps.exe --hierp [--ves] [--ltr 4] [--trust 1] [--refs 16] [--gop 60] [--fps 30] [--frames 300] [--input in.nv12] [--out ltr.h264] [-p n [--other-hierp]]
//   (in.nv12: raw 1280x720 NV12 frames, e.g. ffmpeg -i clip.mp4 -vf scale=1280:720 -pix_fmt nv12 -frames:v 300 -f rawvideo in.nv12)
// Prints max_num_ref_frames from nvEncGetSequenceParams and from the stream. Check the stream with FFmpeg:
//   ffmpeg -i ltr.h264 -f null -
//   ffmpeg -i ltr.h264 -c copy -bsf:v trace_headers -f null - 2>&1 | findstr "max_num_ref_frames long_term_reference_flag memory_management"
// Expected: max_num_ref_frames covers the references the stream uses (maxNumRefFrames, or at least short-term + long-term), or
// nvEncInitializeEncoder refuses the settings.
#define NOMINMAX
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <thread>
#include <vector>

extern "C" {
#include <ffnvcodec/dynlink_loader.h>
}

static thread_local std::string g_tag;
#define STEP(...) do { char b_[512]; snprintf(b_, sizeof b_, __VA_ARGS__); printf("%s%s\n", g_tag.c_str(), b_); fflush(stdout); } while (0)
#define CHECK(call, what) do { NVENCSTATUS s_ = (call); if (s_ != NV_ENC_SUCCESS) { STEP("%s failed: status %d", what, (int)s_); return 1; } } while (0)

// max_num_ref_frames of the first SPS in an Annex B buffer; -1 if it cannot be read.
static int h264_sps_max_refs(const uint8_t* p, size_t n) {
	for (size_t i = 0; i + 4 < n; i++) {
		if (!(p[i] == 0 && p[i + 1] == 0 && p[i + 2] == 1) || (p[i + 3] & 0x1f) != 7) continue;
		std::vector<uint8_t> rb; // the SPS's payload without emulation prevention bytes
		for (size_t j = i + 4, zeros = 0; j < n; j++) {
			if (zeros >= 2 && p[j] == 1) break; // the next start code
			if (zeros >= 2 && p[j] == 3) { zeros = 0; continue; }
			zeros = p[j] == 0 ? zeros + 1 : 0;
			rb.push_back(p[j]);
		}
		size_t bit = 0;
		bool bad = false;
		auto u = [&](int k) {
			uint32_t v = 0;
			for (int x = 0; x < k; x++, bit++) {
				if (bit / 8 >= rb.size()) { bad = true; return 0u; }
				v = (v << 1) | ((rb[bit / 8] >> (7 - bit % 8)) & 1);
			}
			return v;
		};
		auto ue = [&] {
			int lz = 0;
			while (!bad && u(1) == 0) if (++lz > 31) { bad = true; return 0u; }
			return lz ? (1u << lz) - 1 + u(lz) : 0u;
		};
		const uint32_t profile = u(8);
		u(16); // constraint flags, level
		ue(); // seq_parameter_set_id
		if (profile == 100 || profile == 110 || profile == 122 || profile == 244 || profile == 44 || profile == 83 || profile == 86 || profile == 118 || profile == 128 ||
		    profile == 138 || profile == 139 || profile == 134 || profile == 135) {
			if (ue() == 3) u(1); // chroma_format_idc, separate_colour_plane_flag
			ue(); ue(); u(1); // bit depths, qpprime_y_zero_transform_bypass_flag
			if (u(1)) return -1; // scaling matrices: not read
		}
		ue(); // log2_max_frame_num_minus4
		const uint32_t poc = ue();
		if (poc == 0) ue();
		else if (poc == 1) {
			u(1); ue(); ue(); // (se() has the length of ue())
			const uint32_t cycle = ue();
			if (cycle > 255) return -1;
			for (uint32_t c = 0; c < cycle; c++) ue();
		}
		const uint32_t refs = ue();
		return bad ? -1 : (int)refs;
	}
	return -1;
}

struct Opt {
	int ltr = 4, trust = 1, refs = 16, frames = 300, gop = 300, fps = 60;
	std::string out = "ltr.h264", input; // input: raw NV12 1280x720 frames (ffmpeg -i clip.mp4 -vf scale=1280:720 -pix_fmt nv12 -f rawvideo in.nv12)
	bool hierp = false; // NV_ENC_CONFIG_H264::hierarchicalPFrames = 1: the trigger
	bool ves = false; // what ves sets besides: chroma/bit depth fields, VUI (limited range, BT.709 primaries and matrix, sRGB curve), the rest at 0
};

static int run_one(const Opt& op) {
	const int ltr = op.ltr, trust = op.trust, refs = op.refs, frames = op.frames;
	const std::string out = op.out;
	const int w = 1280, h = 720;
	FILE* in = op.input.empty() ? nullptr : fopen(op.input.c_str(), "rb");
	if (!op.input.empty() && !in) { STEP("cannot read %s", op.input.c_str()); return 1; }

	CudaFunctions* cu = nullptr;
	NvencFunctions* nv = nullptr;
	if (cuda_load_functions(&cu, nullptr) < 0 || nvenc_load_functions(&nv, nullptr) < 0) { printf("cannot load CUDA / NVENC\n"); return 1; }
	CUdevice dev;
	CUcontext ctx;
	if (cu->cuInit(0) || cu->cuDeviceGet(&dev, 0) || cu->cuCtxCreate(&ctx, 0, dev)) { printf("CUDA init failed\n"); return 1; }
	char name[128] = {};
	cu->cuDeviceGetName(name, sizeof name, dev);
	int drv = 0;
	cu->cuDriverGetVersion(&drv);
	STEP("GPU: %s (CUDA driver API %d)", name, drv);

	NV_ENCODE_API_FUNCTION_LIST f{};
	f.version = NV_ENCODE_API_FUNCTION_LIST_VER;
	CHECK(nv->NvEncodeAPICreateInstance(&f), "NvEncodeAPICreateInstance");
	NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS os{};
	os.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;
	os.deviceType = NV_ENC_DEVICE_TYPE_CUDA;
	os.device = ctx;
	os.apiVersion = NVENCAPI_VERSION;
	void* enc = nullptr;
	CHECK(f.nvEncOpenEncodeSessionEx(&os, &enc), "nvEncOpenEncodeSessionEx");

	NV_ENC_PRESET_CONFIG pc{};
	pc.version = NV_ENC_PRESET_CONFIG_VER;
	pc.presetCfg.version = NV_ENC_CONFIG_VER;
	CHECK(f.nvEncGetEncodePresetConfigEx(enc, NV_ENC_CODEC_H264_GUID, NV_ENC_PRESET_P4_GUID, NV_ENC_TUNING_INFO_HIGH_QUALITY, &pc), "nvEncGetEncodePresetConfigEx");
	NV_ENC_CONFIG cfg = pc.presetCfg;
	cfg.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;
	cfg.gopLength = (uint32_t)op.gop;
	cfg.frameIntervalP = 1; // no B-frames
	cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
	cfg.rcParams.averageBitRate = cfg.rcParams.maxBitRate = 3500000;
	cfg.rcParams.enableLookahead = 0;
	cfg.rcParams.lookaheadDepth = 0;
	cfg.rcParams.enableAQ = 0;
	cfg.rcParams.enableTemporalAQ = 0;
	cfg.rcParams.multiPass = NV_ENC_MULTI_PASS_DISABLED;
	NV_ENC_CONFIG_H264& c = cfg.encodeCodecConfig.h264Config;
	c.idrPeriod = cfg.gopLength;
	c.useBFramesAsRef = NV_ENC_BFRAME_REF_MODE_DISABLED;
	c.maxNumRefFrames = (uint32_t)refs;
	c.enableLTR = ltr > 0;   // the settings under test
	c.ltrTrustMode = trust;
	c.ltrNumFrames = (uint32_t)ltr;
	if (op.ves) {
		c.chromaFormatIDC = 1;
		c.inputBitDepth = c.outputBitDepth = NV_ENC_BIT_DEPTH_8;
		auto& v = c.h264VUIParameters;
		v.videoSignalTypePresentFlag = 1;
		v.videoFormat = NV_ENC_VUI_VIDEO_FORMAT_UNSPECIFIED;
		v.videoFullRangeFlag = 0;
		v.colourDescriptionPresentFlag = 1;
		v.colourPrimaries = NV_ENC_VUI_COLOR_PRIMARIES_BT709;
		v.transferCharacteristics = NV_ENC_VUI_TRANSFER_CHARACTERISTIC_SRGB;
		v.colourMatrix = NV_ENC_VUI_MATRIX_COEFFS_BT709;
		cfg.rcParams.disableIadapt = cfg.rcParams.disableBadapt = 0;
		cfg.rcParams.strictGOPTarget = 0;
		cfg.rcParams.lookaheadLevel = NV_ENC_LOOKAHEAD_LEVEL_0;
		c.hierarchicalPFrames = 0;
		c.tfLevel = NV_ENC_TEMPORAL_FILTER_LEVEL_0;
	}
	if (op.hierp) c.hierarchicalPFrames = 1;
	STEP("gop %d (idrPeriod too), %d fps, %s input%s%s", op.gop, op.fps, in ? op.input.c_str() : "synthetic", op.ves ? ", ves extras (chroma/depth fields, VUI, explicit zeros)" : "", op.hierp ? ", hierarchicalPFrames 1" : "");
	STEP("h264 %dx%d, P4 hq, CBR 3500 kbps, no B-frames: enableLTR %u, ltrNumFrames %u, ltrTrustMode %u, maxNumRefFrames %u", w, h, (unsigned)c.enableLTR,
	     (unsigned)c.ltrNumFrames, (unsigned)c.ltrTrustMode, (unsigned)c.maxNumRefFrames);

	NV_ENC_INITIALIZE_PARAMS ip{};
	ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
	ip.encodeGUID = NV_ENC_CODEC_H264_GUID;
	ip.presetGUID = NV_ENC_PRESET_P4_GUID;
	ip.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
	ip.encodeWidth = ip.darWidth = w;
	ip.encodeHeight = ip.darHeight = h;
	ip.frameRateNum = (uint32_t)op.fps;
	ip.frameRateDen = 1;
	ip.enablePTD = 1;
	ip.encodeConfig = &cfg;
	CHECK(f.nvEncInitializeEncoder(enc, &ip), "nvEncInitializeEncoder");
	STEP("nvEncInitializeEncoder ok");

	std::vector<uint8_t> hdr(1024);
	uint32_t got = 0;
	NV_ENC_SEQUENCE_PARAM_PAYLOAD sp{};
	sp.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;
	sp.inBufferSize = (uint32_t)hdr.size();
	sp.spsppsBuffer = hdr.data();
	sp.outSPSPPSPayloadSize = &got;
	CHECK(f.nvEncGetSequenceParams(enc, &sp), "nvEncGetSequenceParams");
	const int sps_refs = h264_sps_max_refs(hdr.data(), got);
	STEP("SPS from nvEncGetSequenceParams: max_num_ref_frames = %d (asked: maxNumRefFrames %d, %d long-term)", sps_refs, refs, ltr);

	NV_ENC_CREATE_INPUT_BUFFER ci{};
	ci.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
	ci.width = w;
	ci.height = h;
	ci.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
	CHECK(f.nvEncCreateInputBuffer(enc, &ci), "nvEncCreateInputBuffer");
	NV_ENC_CREATE_BITSTREAM_BUFFER cb{};
	cb.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
	CHECK(f.nvEncCreateBitstreamBuffer(enc, &cb), "nvEncCreateBitstreamBuffer");
	FILE* o = fopen(out.c_str(), "wb");
	if (!o) { printf("cannot write %s\n", out.c_str()); return 1; }

	// Moving content with a cut every 40 frames back to one of three scenes, so that long-term references pay off.
	uint32_t rng = 1;
	std::vector<int> stream_refs;
	for (int i = 0; i < frames; i++) {
		NV_ENC_LOCK_INPUT_BUFFER li{};
		li.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
		li.inputBuffer = ci.inputBuffer;
		CHECK(f.nvEncLockInputBuffer(enc, &li), "nvEncLockInputBuffer");
		uint8_t* d = (uint8_t*)li.bufferDataPtr;
		if (in) {
			bool ok = true;
			for (int y = 0; y < h * 3 / 2 && ok; y++) ok = fread(d + (size_t)y * li.pitch, 1, w, in) == (size_t)w;
			f.nvEncUnlockInputBuffer(enc, ci.inputBuffer);
			if (!ok) { STEP("the input ended after %d frames", i); break; }
			goto submit;
		}
		{
		const int scene = (i / 40) % 3, t = i % 40;
		for (int y = 0; y < h; y++)
			for (int x = 0; x < w; x++) {
				rng = rng * 1664525u + 1013904223u;
				const int v = scene == 0 ? ((x + t * 6) ^ y) & 255 : scene == 1 ? ((x * y) / 64 + t * 3) & 255 : (((x / 32 + y / 32 + t / 4) & 1) ? 200 : 40);
				d[(size_t)y * li.pitch + x] = (uint8_t)std::min(255, std::max(0, v + (int)(rng >> 29) - 4));
			}
		for (int y = 0; y < h / 2; y++)
			for (int x = 0; x < w; x++) d[(size_t)(h + y) * li.pitch + x] = (uint8_t)(128 + ((x / 2 + y + scene * 50 + t) & 63) - 32);
		f.nvEncUnlockInputBuffer(enc, ci.inputBuffer);
		}
	submit:
		NV_ENC_PIC_PARAMS pp{};
		pp.version = NV_ENC_PIC_PARAMS_VER;
		pp.inputBuffer = ci.inputBuffer;
		pp.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
		pp.inputWidth = w;
		pp.inputHeight = h;
		pp.inputPitch = li.pitch;
		pp.outputBitstream = cb.bitstreamBuffer;
		pp.pictureStruct = NV_ENC_PIC_STRUCT_FRAME;
		pp.inputTimeStamp = i;
		const NVENCSTATUS st = f.nvEncEncodePicture(enc, &pp);
		if (st != NV_ENC_SUCCESS && st != NV_ENC_ERR_NEED_MORE_INPUT) { printf("nvEncEncodePicture %d failed: status %d\n", i, (int)st); return 1; }
		if (st == NV_ENC_SUCCESS) {
			NV_ENC_LOCK_BITSTREAM lb{};
			lb.version = NV_ENC_LOCK_BITSTREAM_VER;
			lb.outputBitstream = cb.bitstreamBuffer;
			CHECK(f.nvEncLockBitstream(enc, &lb), "nvEncLockBitstream");
			const uint8_t* b = (const uint8_t*)lb.bitstreamBufferPtr;
			if (int r = h264_sps_max_refs(b, lb.bitstreamSizeInBytes); r >= 0) stream_refs.push_back(r);
			fwrite(b, 1, lb.bitstreamSizeInBytes, o);
			f.nvEncUnlockBitstream(enc, cb.bitstreamBuffer);
		}
	}
	NV_ENC_PIC_PARAMS eos{};
	eos.version = NV_ENC_PIC_PARAMS_VER;
	eos.encodePicFlags = NV_ENC_PIC_FLAG_EOS;
	f.nvEncEncodePicture(enc, &eos);
	fclose(o);
	if (in) fclose(in);
	for (int r : stream_refs) STEP("SPS in the stream: max_num_ref_frames = %d", r);
	STEP("wrote %d frames to %s", frames, out.c_str());
	f.nvEncDestroyBitstreamBuffer(enc, cb.bitstreamBuffer);
	f.nvEncDestroyInputBuffer(enc, ci.inputBuffer);
	f.nvEncDestroyEncoder(enc);
	cu->cuCtxDestroy(ctx);
	return 0;
}

// nvenc_h264_ltr_sps.exe [--ltr n] [--trust 0|1] [--refs n] [--frames n] [--gop n] [--fps n] [--input in.nv12] [--ves] [--out file]
//   (defaults: --ltr 4 --trust 1 --refs 16 --frames 300 --gop 300 --fps 60 --out ltr.h264, synthetic pictures)
//   --ves: what ves (video-encoder-sampler) sets as well; with --gop 60 --fps 30 and a real clip it is the failing ves run.
// nvenc_h264_ltr_sps.exe -p <n> [same options]: n sessions at once in this process, each with its own CUDA context: session 0 with the options,
//   sessions 1..n-1 with maxNumRefFrames 1 and no long-term references; each writes P<i>.h264.
int main(int argc, char** argv) {
	Opt op;
	int parallel = 0;
	bool other_hierp = false;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		auto next = [&] { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
		if (a == "-p") parallel = atoi(next().c_str());
		else if (a == "--ltr") op.ltr = atoi(next().c_str());
		else if (a == "--trust") op.trust = atoi(next().c_str());
		else if (a == "--refs") op.refs = atoi(next().c_str());
		else if (a == "--frames") op.frames = atoi(next().c_str());
		else if (a == "--gop") op.gop = atoi(next().c_str());
		else if (a == "--fps") op.fps = atoi(next().c_str());
		else if (a == "--input") op.input = next();
		else if (a == "--out") op.out = next();
		else if (a == "--ves") op.ves = true;
		else if (a == "--hierp") op.hierp = true;
		else if (a == "--other-hierp") other_hierp = true; // the other -p sessions: hierarchical P without LTR
		else { printf("unknown option %s\n", a.c_str()); return 1; }
	}
	if (parallel > 0) {
		std::vector<std::thread> t;
		for (int i = 0; i < parallel; i++)
			t.emplace_back([i, op, other_hierp] {
				g_tag = "[session " + std::to_string(i) + "] ";
				Opt o = op;
				o.out = "P" + std::to_string(i) + ".h264";
				if (i > 0) { o.ltr = 0; o.trust = 0; o.refs = other_hierp ? 16 : 1; o.hierp = other_hierp; }
				run_one(o);
			});
		for (auto& x : t) x.join();
		return 0;
	}
	return run_one(op);
}
