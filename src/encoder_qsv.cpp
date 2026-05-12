// Native Intel QuickSync backend: talks to oneVPL (libvpl, loaded at runtime) directly, no libavcodec.
// Encoder options are the raw oneVPL field names (TargetUsage, GopRefDist, LookAheadDepth, BRefType, ...) of
// mfxInfoMFX, mfxExtCodingOption2 and mfxExtCodingOption3, so everything those expose can be swept.
// Values are numbers (the enum values of the headers), or the words on / off / adaptive for the MFX_CODINGOPTION_* toggles,
// best_quality / balanced / best_speed for TargetUsage and cbr / vbr / cqp / avbr / la / icq / la_icq / la_hrd / qvbr for RateControlMethod.
// shortcut: written against the headers only, no Intel GPU was available to run it on.
#include "encoder.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "formats.h"
#include "options.h"
#include "gpu.h"
#include "repack.h"

#ifdef _WIN32
#include <windows.h>
#define LIB_HANDLE HMODULE
#define LIB_OPEN(n) LoadLibraryA(n)
#define LIB_SYM(h, n) GetProcAddress(h, n)
#else
#include <dlfcn.h>
#define LIB_HANDLE void*
#define LIB_OPEN(n) dlopen(n, RTLD_NOW)
#define LIB_SYM(h, n) dlsym(h, n)
#endif
#include <vpl/mfx.h>

namespace {

// The oneVPL field names in snake_case; the two that every encoder has under the same name are called that: TargetUsage is preset, CodecProfile is profile, CodecLevel is level.
std::string qsv_name(const char* field) {
	const std::string n = snake_case(field);
	return n == "target_usage" ? "preset" : n == "codec_profile" ? "profile" : n == "codec_level" ? "level" : n;
}

struct Api {
	LIB_HANDLE dll = nullptr;
	bool ok = false;
#define QSV_FUNCS(X) X(MFXLoad) X(MFXUnload) X(MFXCreateConfig) X(MFXSetConfigFilterProperty) X(MFXCreateSession) X(MFXClose) \
	X(MFXVideoENCODE_Query) X(MFXVideoENCODE_QueryIOSurf) X(MFXVideoENCODE_Init) X(MFXVideoENCODE_GetVideoParam) \
	X(MFXVideoENCODE_EncodeFrameAsync) X(MFXVideoENCODE_Close) X(MFXVideoCORE_SyncOperation)
#define X(f) decltype(&::f) f = nullptr;
	QSV_FUNCS(X)
#undef X

	bool load() {
#ifdef _WIN32
		for (const char* name : {"libvpl-2.dll", "libvpl.dll"}) if ((dll = LIB_OPEN(name))) break;
#else
		for (const char* name : {"libvpl.so.2", "libvpl.so"}) if ((dll = LIB_OPEN(name))) break;
#endif
		if (!dll) return false;
#define X(f) if (!(f = (decltype(&::f))LIB_SYM(dll, #f))) return false;
		QSV_FUNCS(X)
#undef X
		return ok = true;
	}
};

// shortcut: the DLL stays loaded for the life of the process, like the AMF runtime
const Api& shared_api() { static Api a = [] { Api x; x.load(); return x; }(); return a; }

struct Surface {
	std::vector<uint8_t> buf;
	mfxFrameSurface1 s{};
};

class Qsv : public Encoder {
	Api a;
	mfxLoader loader = nullptr;
	mfxSession session = nullptr;
	bool open_encoder = false;
	mfxVideoParam par{};
	mfxExtCodingOption2 co2{};
	mfxExtCodingOption3 co3{};
	mfxExtVideoSignalInfo vsi{};
	mfxExtBuffer* ext[3]{};
	std::vector<uint8_t> bits;
	mfxBitstream bs{};
	std::deque<Surface> pool; // stable addresses; a surface is free when the runtime no longer holds a lock on it
	Sink sink;
	int w = 0, h = 0, depth = 8, chroma = 420, pitch = 0, b_delay = 0;
	bool eof = false;

	struct Field { void* p; int size; };
	std::map<std::string, Field> fields;

	bool fail(mfxStatus r, const char* what) {
		fprintf(stderr, "qsv: %s failed (status %d)\n", what, (int)r);
		return false;
	}

	static bool parse(const std::string& v, long long& out) {
		static const std::map<std::string, int> words = {{"on", MFX_CODINGOPTION_ON}, {"off", MFX_CODINGOPTION_OFF}, {"adaptive", MFX_CODINGOPTION_ADAPTIVE},
			{"best_quality", 1}, {"balanced", 4}, {"best_speed", 7},
			{"cbr", MFX_RATECONTROL_CBR}, {"vbr", MFX_RATECONTROL_VBR}, {"cqp", MFX_RATECONTROL_CQP}, {"avbr", MFX_RATECONTROL_AVBR}, {"la", MFX_RATECONTROL_LA},
			{"icq", MFX_RATECONTROL_ICQ}, {"la_icq", MFX_RATECONTROL_LA_ICQ}, {"la_hrd", MFX_RATECONTROL_LA_HRD}, {"qvbr", MFX_RATECONTROL_QVBR}};
		if (auto it = words.find(v); it != words.end()) { out = it->second; return true; }
		char* end = nullptr;
		out = strtoll(v.c_str(), &end, 0);
		return end && end != v.c_str() && !*end;
	}

	bool set(const std::string& key, const std::string& val) {
		auto f = fields.find(key);
		if (f == fields.end()) { fprintf(stderr, "qsv: unknown option '%s'\n", key.c_str()); return false; }
		long long n;
		if (!parse(val, n)) { fprintf(stderr, "qsv: bad value '%s' for '%s'\n", val.c_str(), key.c_str()); return false; }
		switch (f->second.size) {
		case 1: *(uint8_t*)f->second.p = (uint8_t)n; break;
		case 2: *(uint16_t*)f->second.p = (uint16_t)n; break;
		default: *(uint32_t*)f->second.p = (uint32_t)n; break;
		}
		return true;
	}

	void make_fields() {
#define M(n) fields[qsv_name(#n)] = {&par.mfx.n, (int)sizeof(par.mfx.n)};
		M(TargetUsage) M(GopRefDist) M(NumRefFrame) M(IdrInterval) M(NumSlice) M(RateControlMethod) M(MaxKbps) M(InitialDelayInKB)
		M(BufferSizeInKB) M(CodecProfile) M(CodecLevel) M(LowPower) M(QPI) M(QPP) M(QPB) M(ICQQuality) M(Convergence) M(Accuracy) M(GopOptFlag)
#undef M
#define M(n) fields[qsv_name(#n)] = {&co2.n, (int)sizeof(co2.n)};
		M(IntRefType) M(IntRefCycleSize) M(MaxFrameSize) M(MBBRC) M(ExtBRC) M(LookAheadDepth) M(Trellis) M(BRefType) M(AdaptiveI) M(AdaptiveB)
		M(LookAheadDS) M(MinQPI) M(MaxQPI) M(MinQPP) M(MaxQPP) M(MinQPB) M(MaxQPB) M(DisableDeblockingIdc)
#undef M
#define M(n) fields[qsv_name(#n)] = {&co3.n, (int)sizeof(co3.n)};
		M(WinBRCMaxAvgKbps) M(WinBRCSize) M(QVBRQuality) M(WeightedPred) M(WeightedBiPred) M(ScenarioInfo) M(ContentInfo) M(PRefType) M(GPB)
		M(EnableQPOffset) M(LowDelayBRC) M(AdaptiveRef) M(AdaptiveLTR) M(AdaptiveCQM) M(FadeDetection) M(MaxFrameSizeI) M(MaxFrameSizeP)
#undef M
	}

	mfxFrameSurface1* acquire() {
		for (auto& p : pool) if (!p.s.Data.Locked) return &p.s;
		const int alignedH = (h + 15) & ~15;
		Surface& p = pool.emplace_back();
		p.s.Info = par.mfx.FrameInfo;
		p.buf.assign((size_t)pitch * alignedH * (chroma == 444 ? 2 : 3) / 2, 0); // 4:4:4 is packed, 4:2:0 is a plane and a half
		mfxFrameData& d = p.s.Data;
		d.PitchLow = pitch & 0xFFFF; d.PitchHigh = (mfxU16)(pitch >> 16);
		if (chroma == 444) {
			if (depth > 8) d.Y410 = (mfxY410*)p.buf.data();
			else d.V = p.buf.data(); // AYUV is V, U, Y, A in memory
		} else {
			d.Y = p.buf.data();
			d.UV = p.buf.data() + (size_t)pitch * alignedH;
		}
		return &p.s;
	}

	void fill(const Picture* fr, mfxFrameSurface1* s) const {
		const int sample = depth > 8 ? 2 : 1;
		auto get = [&](int plane, int row, int x) {
			const uint8_t* line = fr->data[plane] + (size_t)row * fr->stride[plane];
			return sample == 1 ? (int)line[x] : (int)((const uint16_t*)line)[x];
		};
		mfxFrameData& d = s->Data;
		if (chroma == 444) {
			for (int row = 0; row < h; row++) for (int x = 0; x < w; x++) {
				const int y = get(0, row, x), u = get(1, row, x), v = get(2, row, x);
				if (depth > 8) ((uint32_t*)((uint8_t*)d.Y410 + (size_t)row * pitch))[x] = (uint32_t)u | (uint32_t)y << 10 | (uint32_t)v << 20 | 3u << 30;
				else { uint8_t* px = d.V + (size_t)row * pitch + 4 * x; px[0] = (uint8_t)v; px[1] = (uint8_t)u; px[2] = (uint8_t)y; px[3] = 255; }
			}
		} else {
			pack_plane(*fr, 0, w, h, depth, d.Y, pitch);
			pack_uv(*fr, w, h, depth, d.UV, pitch);
		}
		d.TimeStamp = (mfxU64)fr->pts;
	}

	bool emit(mfxSyncPoint sp) {
		mfxStatus r = a.MFXVideoCORE_SyncOperation(session, sp, 60000);
		if (r < 0) return fail(r, "SyncOperation");
		stats.packets++; stats.bytes += bs.DataLength;
		if (sink) sink(bs.Data + bs.DataOffset, bs.DataLength, (int64_t)bs.TimeStamp, (bs.FrameType & MFX_FRAMETYPE_IDR) != 0, NO_DTS);
		bs.DataOffset = 0; bs.DataLength = 0;
		return true;
	}

public:
	~Qsv() override {
		if (open_encoder) a.MFXVideoENCODE_Close(session);
		if (session) a.MFXClose(session);
		if (loader) a.MFXUnload(loader);
	}

	bool open(const Params& p, Sink s) override {
		sink = std::move(s);
		w = p.width; h = p.height;
		const Fmt fmt = p.pix;
		depth = fmt.depth; chroma = fmt.chroma;
		if ((chroma != 420 && chroma != 444) || depth > 10) { fprintf(stderr, "qsv: cannot encode %d-bit %d\n", depth, chroma); return false; }
		const mfxU32 codec = p.codec == "h264" ? MFX_CODEC_AVC : p.codec == "hevc" ? MFX_CODEC_HEVC : p.codec == "av1" ? MFX_CODEC_AV1 : 0;
		if (!codec) { fprintf(stderr, "qsv: unknown codec '%s'\n", p.codec.c_str()); return false; }
		Options o = p.options;
		int gpu = 0;
		if (auto it = o.find("gpu"); it != o.end()) { gpu = atoi(it->second.c_str()); o.erase(it); }

		if (list_gpus(VENDOR_INTEL).size() <= (size_t)gpu) { fprintf(stderr, "qsv: no Intel GPU with index %d\n", gpu); return false; }
		a = shared_api();
		if (!a.ok) { fprintf(stderr, "qsv: libvpl not found\n"); return false; }

		// Only Intel hardware implementations that can encode this codec; the gpu option picks among those.
		if (!(loader = a.MFXLoad())) { fprintf(stderr, "qsv: cannot create the oneVPL loader\n"); return false; }
		auto filter = [&](const char* name, mfxU32 value) {
			mfxConfig cfg = a.MFXCreateConfig(loader);
			mfxVariant v{};
			v.Version.Version = MFX_VARIANT_VERSION;
			v.Type = MFX_VARIANT_TYPE_U32;
			v.Data.U32 = value;
			return cfg && a.MFXSetConfigFilterProperty(cfg, (const mfxU8*)name, v) == MFX_ERR_NONE;
		};
		if (!filter("mfxImplDescription.Impl", MFX_IMPL_TYPE_HARDWARE) || !filter("mfxImplDescription.VendorID", VENDOR_INTEL) ||
		    !filter("mfxImplDescription.mfxEncoderDescription.encoder.CodecID", codec)) { fprintf(stderr, "qsv: cannot set the oneVPL filters\n"); return false; }
		mfxStatus r = a.MFXCreateSession(loader, gpu, &session);
		if (r != MFX_ERR_NONE) { fprintf(stderr, "qsv: no Intel GPU %d that encodes %s (status %d)\n", gpu, p.codec.c_str(), (int)r); return false; }

		par.IOPattern = MFX_IOPATTERN_IN_SYSTEM_MEMORY;
		par.AsyncDepth = 1;
		mfxInfoMFX& m = par.mfx;
		m.CodecId = codec;
		m.TargetUsage = 4;
		m.GopPicSize = (mfxU16)p.gop;
		m.RateControlMethod = MFX_RATECONTROL_CBR;
		m.BRCParamMultiplier = (mfxU16)std::max<int64_t>(1, (p.bitrate / 1000 + 64999) / 65000); // the kbps fields are 16 bit
		m.TargetKbps = m.MaxKbps = (mfxU16)(p.bitrate / 1000 / m.BRCParamMultiplier);
		if (codec == MFX_CODEC_AV1) m.LowPower = MFX_CODINGOPTION_ON; // AV1 only exists on the fixed-function encoder
		if (codec == MFX_CODEC_HEVC) m.CodecProfile = chroma == 444 ? MFX_PROFILE_HEVC_REXT : depth > 8 ? MFX_PROFILE_HEVC_MAIN10 : MFX_PROFILE_HEVC_MAIN;
		if (codec == MFX_CODEC_AV1) m.CodecProfile = MFX_PROFILE_AV1_MAIN;

		mfxFrameInfo& fi = m.FrameInfo;
		fi.FourCC = chroma == 444 ? (depth > 8 ? MFX_FOURCC_Y410 : MFX_FOURCC_AYUV) : depth > 8 ? MFX_FOURCC_P010 : MFX_FOURCC_NV12;
		fi.ChromaFormat = chroma == 444 ? MFX_CHROMAFORMAT_YUV444 : MFX_CHROMAFORMAT_YUV420;
		fi.BitDepthLuma = fi.BitDepthChroma = (mfxU16)depth;
		fi.Shift = chroma == 420 && depth > 8; // P010 keeps its samples in the top bits
		fi.PicStruct = MFX_PICSTRUCT_PROGRESSIVE;
		fi.Width = (mfxU16)((w + 15) & ~15); fi.Height = (mfxU16)((h + 15) & ~15);
		fi.CropW = (mfxU16)w; fi.CropH = (mfxU16)h;
		fi.FrameRateExtN = p.framerate.num; fi.FrameRateExtD = p.framerate.den;
		pitch = fi.Width * (chroma == 444 ? 4 : depth > 8 ? 2 : 1);

		co2.Header.BufferId = MFX_EXTBUFF_CODING_OPTION2; co2.Header.BufferSz = sizeof co2;
		co3.Header.BufferId = MFX_EXTBUFF_CODING_OPTION3; co3.Header.BufferSz = sizeof co3;
		ext[0] = &co2.Header; ext[1] = &co3.Header;
		par.ExtParam = ext; par.NumExtParam = 2;
		if (color_full(p.color) || color_described(p.color)) { // the colour of the pictures, written into the stream
			vsi.Header.BufferId = MFX_EXTBUFF_VIDEO_SIGNAL_INFO; vsi.Header.BufferSz = sizeof vsi;
			vsi.VideoFormat = 5; // unspecified
			vsi.VideoFullRange = color_full(p.color);
			if (color_described(p.color)) {
				vsi.ColourDescriptionPresent = 1;
				vsi.ColourPrimaries = (mfxU16)p.color.primaries; vsi.TransferCharacteristics = (mfxU16)p.color.trc; vsi.MatrixCoefficients = (mfxU16)p.color.space;
			}
			ext[2] = &vsi.Header; par.NumExtParam = 3;
		}

		make_fields();
		NativeOptions native;
		std::string why;
		if (!resolve_options("qsv", p.codec, o, [](const std::string& k) { return k; }, native, why)) { fprintf(stderr, "qsv: %s\n", why.c_str()); return false; }
		for (auto& [k, v] : native) if (!set(k, v)) return false;

		r = a.MFXVideoENCODE_Init(session, &par);
		if (r < 0) return fail(r, "Init");
		if (r > 0) fprintf(stderr, "qsv: the runtime adjusted some parameters (status %d)\n", (int)r);
		open_encoder = true;

		mfxVideoParam got{};
		got.mfx.CodecId = codec;
		if ((r = a.MFXVideoENCODE_GetVideoParam(session, &got)) < 0) return fail(r, "GetVideoParam");
		b_delay = (int)got.mfx.GopRefDist > 1 ? (int)got.mfx.GopRefDist - 1 : 0; // the runtime reports the distance it settled on
		bits.resize((size_t)std::max<mfxU32>(got.mfx.BufferSizeInKB, 1024) * std::max<mfxU16>(got.mfx.BRCParamMultiplier, 1) * 1000);
		bs.Data = bits.data(); bs.MaxLength = (mfxU32)bits.size();
		return true;
	}

	int reorder_delay() const override { return b_delay; }

	bool encode(const Picture* fr) override {
		mfxFrameSurface1* s = nullptr;
		if (fr) { s = acquire(); fill(fr, s); }
		for (;;) {
			mfxSyncPoint sp = nullptr;
			mfxStatus r = a.MFXVideoENCODE_EncodeFrameAsync(session, nullptr, s, &bs, &sp);
			if (r == MFX_WRN_DEVICE_BUSY) { std::this_thread::sleep_for(std::chrono::milliseconds(1)); continue; }
			if (r == MFX_ERR_MORE_DATA) return true; // input taken, nothing finished yet; when draining: all done
			if (r < 0) return fail(r, "EncodeFrameAsync");
			if (sp && !emit(sp)) return false;
			if (s) return true; // one output at most per input
		}
	}
};

} // namespace

std::unique_ptr<Encoder> make_encoder_qsv() { return std::make_unique<Qsv>(); }

// Whether this machine's Intel GPU can encode the codec in this layout, found by initializing a throwaway encoder once.
bool qsv_supports(const std::string& codec, int chroma, int depth) {
	if ((chroma != 420 && chroma != 444) || depth > 10) return false;
	static std::mutex m;
	static std::map<std::string, bool> cache;
	std::lock_guard<std::mutex> lock(m);
	const std::string key = codec + std::to_string(chroma) + std::to_string(depth);
	auto it = cache.find(key);
	if (it == cache.end()) {
		Params p;
		p.width = 1280; p.height = 720; p.bitrate = 5000000; p.gop = 60;
		p.pix = planar_fmt(chroma, depth > 8 ? 10 : 8);
		p.codec = codec;
		it = cache.emplace(key, make_encoder_qsv()->open(p, nullptr)).first;
	}
	return it->second;
}
