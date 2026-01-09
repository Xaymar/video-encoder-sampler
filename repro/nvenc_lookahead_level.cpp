// Reproduces a process abort (exception 0xC0000409, fast-fail) inside NVIDIA's user-mode encoder library:
// NV_ENC_RC_PARAMS::lookaheadLevel is set to a non-zero value while NV_ENC_RC_PARAMS::enableLookahead is 0.
// The same configuration with lookaheadLevel = 0, or with enableLookahead = 1 (and lookaheadDepth > 0), encodes normally.
//
// Uses only the public NVENC API through FFmpeg's ffnvcodec headers (nv-codec-headers), which load nvcuda.dll and nvEncodeAPI64.dll at run time.
//
// Build (Windows, clang; nv-codec-headers' include folder is the only dependency):
//   clang++ -std=c++17 -I <nv-codec-headers>/include nvenc_lookahead_level.cpp -o nvenc_lookahead_level.exe
// Run:
//   nvenc_lookahead_level.exe [lookaheadLevel] [codec: hevc | h264 | av1] [enableLookahead: 0 | 1]
//   nvenc_lookahead_level.exe 2 hevc 0     aborts
//   nvenc_lookahead_level.exe 0 hevc 0     encodes 10 frames
//   nvenc_lookahead_level.exe 2 hevc 1     encodes 10 frames
//
// Observed (GeForce RTX 4080 SUPER, Windows 11, driver 32.0.16.1742 = 617.42, nvEncodeAPI64.dll / nvcuvid.dll 7.17.16.1742,
// nv-codec-headers with NVENCAPI_MAJOR_VERSION 13):
//   lookaheadLevel 0, any codec                     encodes
//   lookaheadLevel 1..3, enableLookahead 1, hevc    encodes
//   lookaheadLevel 1..3, enableLookahead 0, h264    encodes (H.264 has no lookahead level)
//   lookaheadLevel 1..3, enableLookahead 0, hevc/av1  nvEncInitializeEncoder succeeds, then the first nvEncEncodePicture ends the process
//                                                   with exception 0xC0000409 inside nvcuvid.dll, instead of returning an error status.
// Expected: nvEncInitializeEncoder (or nvEncEncodePicture) returns an error such as NV_ERR_INVALID_PARAM, or the setting is ignored.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include <ffnvcodec/dynlink_loader.h>
}

#define STEP(...) do { printf(__VA_ARGS__); printf("\n"); fflush(stdout); } while (0)
#define CHECK(call, what) do { NVENCSTATUS s_ = (call); if (s_ != NV_ENC_SUCCESS) { printf("%s failed: status %d\n", what, (int)s_); return 1; } } while (0)

int main(int argc, char** argv) {
	const int level = argc > 1 ? atoi(argv[1]) : 2;
	const std::string codec = argc > 2 ? argv[2] : "hevc";
	const int lookahead = argc > 3 ? atoi(argv[3]) : 0;
	const int w = 1280, h = 720, frames = 10;

	CudaFunctions* cu = nullptr;
	NvencFunctions* nv = nullptr;
	if (cuda_load_functions(&cu, nullptr) < 0 || nvenc_load_functions(&nv, nullptr) < 0) { printf("cannot load CUDA / NVENC\n"); return 1; }
	CUdevice dev;
	CUcontext ctx;
	if (cu->cuInit(0) || cu->cuDeviceGet(&dev, 0) || cu->cuCtxCreate(&ctx, 0, dev)) { printf("CUDA init failed\n"); return 1; }
	char name[128] = {};
	cu->cuDeviceGetName(name, sizeof name, dev);
	STEP("GPU: %s", name);

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

	const GUID guid = codec == "h264" ? NV_ENC_CODEC_H264_GUID : codec == "av1" ? NV_ENC_CODEC_AV1_GUID : NV_ENC_CODEC_HEVC_GUID;
	NV_ENC_PRESET_CONFIG pc{};
	pc.version = NV_ENC_PRESET_CONFIG_VER;
	pc.presetCfg.version = NV_ENC_CONFIG_VER;
	CHECK(f.nvEncGetEncodePresetConfigEx(enc, guid, NV_ENC_PRESET_P4_GUID, NV_ENC_TUNING_INFO_HIGH_QUALITY, &pc), "nvEncGetEncodePresetConfigEx");
	NV_ENC_CONFIG cfg = pc.presetCfg;
	cfg.gopLength = 60;
	cfg.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;
	cfg.rcParams.averageBitRate = cfg.rcParams.maxBitRate = 2000000;

	cfg.rcParams.enableLookahead = lookahead ? 1 : 0;
	cfg.rcParams.lookaheadDepth = lookahead ? 16 : 0;
	cfg.rcParams.lookaheadLevel = (NV_ENC_LOOKAHEAD_LEVEL)level; // the setting under test
	STEP("codec %s, enableLookahead %u, lookaheadDepth %u, lookaheadLevel %d", codec.c_str(), (unsigned)cfg.rcParams.enableLookahead, (unsigned)cfg.rcParams.lookaheadDepth, level);

	NV_ENC_INITIALIZE_PARAMS ip{};
	ip.version = NV_ENC_INITIALIZE_PARAMS_VER;
	ip.encodeGUID = guid;
	ip.presetGUID = NV_ENC_PRESET_P4_GUID;
	ip.tuningInfo = NV_ENC_TUNING_INFO_HIGH_QUALITY;
	ip.encodeWidth = ip.darWidth = w;
	ip.encodeHeight = ip.darHeight = h;
	ip.frameRateNum = 30;
	ip.frameRateDen = 1;
	ip.enablePTD = 1;
	ip.encodeConfig = &cfg;
	STEP("nvEncInitializeEncoder ...");
	CHECK(f.nvEncInitializeEncoder(enc, &ip), "nvEncInitializeEncoder");
	STEP("nvEncInitializeEncoder ok");

	NV_ENC_CREATE_INPUT_BUFFER ci{};
	ci.version = NV_ENC_CREATE_INPUT_BUFFER_VER;
	ci.width = w;
	ci.height = h;
	ci.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12;
	CHECK(f.nvEncCreateInputBuffer(enc, &ci), "nvEncCreateInputBuffer");
	NV_ENC_CREATE_BITSTREAM_BUFFER cb{};
	cb.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;
	CHECK(f.nvEncCreateBitstreamBuffer(enc, &cb), "nvEncCreateBitstreamBuffer");

	for (int i = 0; i < frames; i++) {
		NV_ENC_LOCK_INPUT_BUFFER li{};
		li.version = NV_ENC_LOCK_INPUT_BUFFER_VER;
		li.inputBuffer = ci.inputBuffer;
		CHECK(f.nvEncLockInputBuffer(enc, &li), "nvEncLockInputBuffer");
		for (int y = 0; y < h * 3 / 2; y++) memset((uint8_t*)li.bufferDataPtr + (size_t)y * li.pitch, y < h ? (y + i * 4) & 255 : 128, w);
		f.nvEncUnlockInputBuffer(enc, ci.inputBuffer);

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
		STEP("nvEncEncodePicture %d ...", i);
		NVENCSTATUS st = f.nvEncEncodePicture(enc, &pp);
		if (st != NV_ENC_SUCCESS && st != NV_ENC_ERR_NEED_MORE_INPUT) { printf("nvEncEncodePicture failed: status %d\n", (int)st); return 1; }
		if (st == NV_ENC_SUCCESS) {
			NV_ENC_LOCK_BITSTREAM lb{};
			lb.version = NV_ENC_LOCK_BITSTREAM_VER;
			lb.outputBitstream = cb.bitstreamBuffer;
			CHECK(f.nvEncLockBitstream(enc, &lb), "nvEncLockBitstream");
			STEP("  got %u bytes", lb.bitstreamSizeInBytes);
			f.nvEncUnlockBitstream(enc, cb.bitstreamBuffer);
		}
	}
	STEP("done: %d frames submitted without an abort", frames);
	f.nvEncDestroyBitstreamBuffer(enc, cb.bitstreamBuffer);
	f.nvEncDestroyInputBuffer(enc, ci.inputBuffer);
	f.nvEncDestroyEncoder(enc);
	return 0;
}
