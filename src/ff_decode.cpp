// Decoder on top of libavcodec, for the encoded streams that are scored.
//
// The software decoder is the truth, but it is only used until a GPU has shown that it decodes a kind of stream (codec, size, bit depth, chroma) exactly
// the same. With hardware decoding on (--hw-decode, on the GPU --decode-gpu picks), a stream of a kind that is not known yet is decoded by the GPU and
// in software side by side, all the way through, and the software frames are the ones handed out. When every frame matched, the GPU is truthful for
// that kind: this is kept on disk (see save_verdict()), and from then on streams of that kind decode on the GPU alone, with no software decoder at all.
// A difference (or the GPU giving fewer frames, a format we cannot use, an error) drops the GPU decoder and software carries on; that kind then stays
// in software on that GPU. VES_HW_DECODE_VERIFY=<frames>|all compares known kinds as well, to look at a machine again.
// Only FFmpeg's own decoders on a hardware device (hwaccel) are used, not the dedicated ones (h264_cuvid, ...): they parse, reorder and crop
// the stream with the same code as the software decoder, and only the reconstruction runs on the GPU. (AV1 is the exception: software is libdav1d,
// the GPU gets FFmpeg's own AV1 decoder, which has no software path of its own; the check covers that as well.)
#include "decode.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <string>

#include "ff.h"
#include "gpu.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
}

namespace {

// VES_HW_DECODE_VERIFY=<frames>|all: compare every stream that much even when the device is known to be truthful for its kind. 0 (unset): rely on the verdicts.
long long verify_frames() {
	if (const char* e = getenv("VES_HW_DECODE_VERIFY")) return strcmp(e, "all") == 0 ? -1 : atoll(e);
	return 0;
}

std::mutex g_mutex;
std::set<std::string> g_bad;          // kind|type|device that differed from software or could not decode in this process
std::set<std::string> g_told;         // messages said once

// Verdicts that outlive the process: "good" once a whole stream of a kind decoded exactly like software on a device, "bad" once it did not. The key
// holds the GPU and its driver and the libavcodec version, so an update of either checks again. One line per verdict, appended, in the user's cache
// folder (VES_HW_DECODE_VERDICTS=<file> to put it elsewhere, "off" for none).
std::map<std::string, bool> g_verdicts;
bool g_verdicts_loaded = false;

std::string verdict_file() {
	if (const char* e = getenv("VES_HW_DECODE_VERDICTS")) return strcmp(e, "off") == 0 ? std::string() : std::string(e);
	std::error_code ec;
	std::filesystem::path dir;
#ifdef _WIN32
	if (const char* a = getenv("LOCALAPPDATA")) dir = std::filesystem::path(a) / "ves";
#else
	if (const char* x = getenv("XDG_CACHE_HOME"); x && *x) dir = std::filesystem::path(x) / "ves";
	else if (const char* h = getenv("HOME")) dir = std::filesystem::path(h) / ".cache" / "ves";
#endif
	if (dir.empty()) return {};
	std::filesystem::create_directories(dir, ec);
	return (dir / "hw-decode-verdicts.txt").string();
}

// With g_mutex held.
void load_verdicts() {
	if (g_verdicts_loaded) return;
	g_verdicts_loaded = true;
	const std::string file = verdict_file();
	if (file.empty()) return;
	std::ifstream in(file);
	for (std::string line; std::getline(in, line);) {
		const size_t tab = line.find('\t');
		if (tab == std::string::npos) continue;
		const std::string what = line.substr(0, tab), key = line.substr(tab + 1);
		if (what == "good" || what == "bad") g_verdicts[key] = what == "good"; // the last line for a key wins
	}
}

// With g_mutex held.
void save_verdict(const std::string& key, bool good) {
	load_verdicts();
	if (auto it = g_verdicts.find(key); it != g_verdicts.end() && it->second == good) return;
	g_verdicts[key] = good;
	const std::string file = verdict_file();
	if (file.empty()) return;
	if (FILE* f = fopen(file.c_str(), "ab")) { // one short line per write: processes that append at the same time do not mix their lines
		fprintf(f, "%s\t%s\n", good ? "good" : "bad", key.c_str());
		fclose(f);
	}
}

void tell_once(const std::string& key, const std::string& msg) {
	{
		std::lock_guard lock(g_mutex);
		if (!g_told.insert(key).second) return;
	}
	fprintf(stderr, "%s\n", msg.c_str());
}

// One libavcodec decoder and the frames it has given that were not handed out yet.
struct Dec {
	AVCodecContext* ctx = nullptr;
	AVPixelFormat hw_fmt = AV_PIX_FMT_NONE; // the GPU's frame format; NONE for software
	AVFrame* frame = av_frame_alloc();
	AVFrame* down = av_frame_alloc();   // a GPU frame read back
	AVFrame* planar = av_frame_alloc(); // and unpacked to planar YUV
	AVBufferRef* dev = nullptr;
	std::deque<Picture> ready;
	bool eof = false;
	long long got = 0; // frames received

	~Dec() {
		av_frame_free(&frame);
		av_frame_free(&down);
		av_frame_free(&planar);
		avcodec_free_context(&ctx);
		av_buffer_unref(&dev);
	}

	static AVPixelFormat pick_format(AVCodecContext* c, const AVPixelFormat* fmts) {
		const AVPixelFormat want = (AVPixelFormat)(intptr_t)c->opaque;
		for (const AVPixelFormat* p = fmts; *p != AV_PIX_FMT_NONE; p++) if (*p == want) return *p;
		return AV_PIX_FMT_NONE; // the GPU cannot do this stream: fail rather than decode in software a second time
	}

	// type empty: software.
	bool open(AVCodecID id, const std::vector<uint8_t>& extradata, const std::string& type, const std::string& device) {
		const AVCodec* c = type.empty() || type == "test" ? software_decoder(id) : hwaccel_decoder(id, type);
		if (!c) return false;
		ctx = avcodec_alloc_context3(c);
		if (!ctx) return false;
		// Without VUI reorder info the H.264 decoder guesses the B-frame depth and can drop or misorder early frames;
		// strict mode makes it assume the maximum, so every frame comes out in display order.
		ctx->strict_std_compliance = FF_COMPLIANCE_STRICT;
		AVDictionary* opts = nullptr;
		if (type == "test") { // VES_HW_DECODE_TEST: a second software decoder plays the GPU, with these decoder options (skip_loop_filter=all makes it differ)
			ctx->thread_count = 1;
			av_dict_parse_string(&opts, getenv("VES_HW_DECODE_TEST"), "=", ":", 0);
		} else if (type.empty()) {
			ctx->thread_count = decode_threads();
			ctx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
		} else {
			const AVHWDeviceType t = av_hwdevice_find_type_by_name(type.c_str());
			if (t == AV_HWDEVICE_TYPE_NONE) return false;
			for (int i = 0;; i++) {
				const AVCodecHWConfig* cfg = avcodec_get_hw_config(c, i);
				if (!cfg) return false;
				if ((cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX) && cfg->device_type == t) { hw_fmt = cfg->pix_fmt; break; }
			}
			if (!(dev = shared_hw_device(type, device))) return false;
			ctx->hw_device_ctx = av_buffer_ref(dev);
			ctx->opaque = (void*)(intptr_t)hw_fmt;
			ctx->get_format = &Dec::pick_format;
			ctx->thread_count = 1; // the GPU does the work
		}
		if (!extradata.empty()) {
			ctx->extradata = (uint8_t*)av_mallocz(extradata.size() + AV_INPUT_BUFFER_PADDING_SIZE);
			if (ctx->extradata) { memcpy(ctx->extradata, extradata.data(), extradata.size()); ctx->extradata_size = (int)extradata.size(); }
		}
		const bool opened = avcodec_open2(ctx, c, &opts) >= 0;
		av_dict_free(&opts);
		return opened;
	}

	bool send(const AVPacket* pkt) { return avcodec_send_packet(ctx, pkt) >= 0; }

	// Moves every frame the decoder has into `ready`. <0: an error (`why` says which), else 0.
	int pump(std::string& why) {
		for (;;) {
			const int r = avcodec_receive_frame(ctx, frame);
			if (r == AVERROR(EAGAIN)) return 0;
			if (r == AVERROR_EOF) { eof = true; return 0; }
			char b[AV_ERROR_MAX_STRING_SIZE];
			if (r < 0) { why = std::string("decoding failed: ") + av_make_error_string(b, sizeof b, r); return r; }
			AVFrame* f = frame;
			if (hw_fmt != AV_PIX_FMT_NONE) {
				if (frame->format != hw_fmt) { av_frame_unref(frame); why = "the GPU did not decode the frame"; return AVERROR(ENOSYS); }
				av_frame_unref(down);
				if (const int t = av_hwframe_transfer_data(down, frame, 0); t < 0) {
					av_frame_unref(frame);
					why = std::string("reading a hardware decoded frame back failed: ") + av_make_error_string(b, sizeof b, t);
					return AVERROR_EXTERNAL;
				}
				av_frame_copy_props(down, frame);
				av_frame_unref(frame);
				f = unpack_semi_planar(down, planar);
			}
			Fmt fmt;
			if (!describe_native((AVPixelFormat)f->format, fmt)) {
				why = std::string("the decoded frames are ") + (av_get_pix_fmt_name((AVPixelFormat)f->format) ? av_get_pix_fmt_name((AVPixelFormat)f->format) : "of an unknown format");
				av_frame_unref(f);
				return AVERROR_INVALIDDATA;
			}
			ready.push_back(picture_from_frame(f, fmt));
			got++;
		}
	}
};

class FfDecoder : public Decoder {
	AVPacket* pkt = av_packet_alloc();
	std::unique_ptr<Dec> sw, hw;
	std::string kind, hw_key, hw_name; // the stream's kind, the hardware decoder's verdict key, and what to call it
	long long check = 0;               // frames still to compare; -1: all of them
	long long compared = 0;

	void drop_hw(const std::string& why) {
		{
			std::lock_guard lock(g_mutex);
			g_bad.insert(hw_key);
			save_verdict(hw_key, false);
		}
		tell_once("drop|" + hw_key, "scored streams: hardware decoding (" + hw_name + ") is not used for " + kind + ": " + why + "; decoding in software");
		hw.reset();
	}

	// The comparison ends with every frame the same; `whole`: it covered the whole stream, which makes the device truthful for this kind.
	void drop_sw(bool whole) {
		if (whole) {
			std::lock_guard lock(g_mutex);
			save_verdict(hw_key, true);
		}
		tell_once("ok|" + hw_key + (whole ? "|all" : ""), "scored streams: hardware decoding (" + hw_name + ") matched software on " +
		          (whole ? "all " + std::to_string(compared) + " frames of a " + kind + " stream; streams of this kind now decode on the GPU alone" : "the first " + std::to_string(compared) + " frames of a " + kind + " stream"));
		sw.reset();
	}

public:
	~FfDecoder() override { av_packet_free(&pkt); }

	bool open(Codec codec, const std::vector<uint8_t>& extradata, const Shape* enc) override {
		const AVCodecID id = codec == Codec::H264 ? AV_CODEC_ID_H264 : codec == Codec::HEVC ? AV_CODEC_ID_HEVC : codec == Codec::AV1 ? AV_CODEC_ID_AV1 : AV_CODEC_ID_NONE;
		if (id == AV_CODEC_ID_NONE) { fprintf(stderr, "no decoder for encoded stream\n"); return false; }
		auto open_sw = [&] {
			sw = std::make_unique<Dec>();
			if (sw->open(id, extradata, "", "")) return true;
			fprintf(stderr, "no decoder for encoded stream\n");
			return false;
		};
		if (!enc) return open_sw(); // no kind to keep verdicts for: software only
		kind = std::string(avcodec_get_name(id)) + " " + std::to_string(enc->w) + "x" + std::to_string(enc->h) + " " + std::to_string(enc->pix.depth) + "-bit " +
		       (enc->pix.chroma == 420 ? "4:2:0" : enc->pix.chroma == 422 ? "4:2:2" : "4:4:4");
		// The first hardware decoder that opens and is not known to be bad for this kind of stream.
		std::vector<HwChoice> choices = hw_decode_choices(id);
		if (getenv("VES_HW_DECODE_TEST")) choices = {{"test", ""}};
		const std::string where = "|" + hw_decode_gpu_identity() + "|lavc " + std::to_string(avcodec_version());
		bool good = false;
		for (const HwChoice& c : choices) {
			const std::string key = kind + "|" + c.type + "|" + c.device + where;
			{
				std::lock_guard lock(g_mutex);
				load_verdicts();
				if (g_bad.count(key)) continue;
				if (auto it = g_verdicts.find(key); it != g_verdicts.end() && !it->second) continue;
			}
			auto d = std::make_unique<Dec>();
			if (!d->open(id, extradata, c.type, c.device)) { // no such device here, or it cannot decode the codec
				std::lock_guard lock(g_mutex);
				g_bad.insert(key);
				continue;
			}
			hw = std::move(d);
			hw_key = key;
			hw_name = c.type + (c.device.empty() ? "" : ", device " + c.device);
			std::lock_guard lock(g_mutex);
			if (auto it = g_verdicts.find(key); it != g_verdicts.end()) good = it->second;
			break;
		}
		if (!hw) {
			if (!choices.empty()) tell_once("none|" + kind, "scored streams: no hardware decoder can be used for " + kind + " streams: decoding in software");
			return open_sw();
		}
		check = verify_frames();
		if (good && check == 0) { // truthful for this kind already: the GPU alone
			tell_once("gpu|" + hw_key, "scored streams: " + kind + " streams decode on the GPU (" + hw_name + "), which matched software on a whole stream of this kind before");
			return true;
		}
		if (!good) check = -1; // not known yet: this stream is compared all the way through
		return open_sw();
	}

	bool send(const uint8_t* data, size_t size) override {
		if (av_new_packet(pkt, (int)size) < 0) return false;
		memcpy(pkt->data, data, size);
		bool ok = true;
		if (sw) ok = sw->send(pkt);
		if (hw && !hw->send(pkt)) {
			if (sw) drop_hw("it refused a packet");
			else ok = false;
		}
		av_packet_unref(pkt);
		return ok;
	}

	bool flush() override {
		bool ok = true;
		if (sw) ok = sw->send(nullptr);
		if (hw && !hw->send(nullptr)) {
			if (sw) drop_hw("it refused the end of the stream");
			else ok = false;
		}
		return ok;
	}

	int receive(Picture& out) override {
		for (;;) {
			std::string why;
			if (hw && hw->pump(why) < 0) {
				if (!sw) { // past the check: there is no software decode to fall back on
					if (why.rfind("reading", 0) == 0) gpu_lost(why);
					fprintf(stderr, "scored streams: hardware decoding (%s) failed after %lld frames: %s\n", hw_name.c_str(), hw->got, why.c_str());
					std::lock_guard lock(g_mutex);
					g_bad.insert(hw_key);
					save_verdict(hw_key, false); // the run fails; when it is tried again, it decodes in software
					return -1;
				}
				drop_hw(why);
			}
			if (sw && sw->pump(why) < 0) { fprintf(stderr, "vmaf: %s\n", why.c_str()); return -1; }
			if (!hw || !sw) { // one decoder
				Dec& d = hw ? *hw : *sw;
				if (d.ready.empty()) return 0;
				out = std::move(d.ready.front());
				d.ready.pop_front();
				return 1;
			}
			if (!sw->ready.empty() && !hw->ready.empty()) { // the check: software's frame goes out, the GPU's is compared with it
				Picture a = std::move(sw->ready.front()), b = std::move(hw->ready.front());
				sw->ready.pop_front();
				hw->ready.pop_front();
				if (a.fmt != b.fmt || !same_pictures(a, b)) {
					fprintf(stderr, "scored streams: hardware decoding (%s) differs from software at frame %lld of a %s stream\n", hw_name.c_str(), compared, kind.c_str());
					describe_difference(a, b, (int)compared);
					drop_hw("its frames differ from software");
				} else if (++compared == check) drop_sw(false);
				out = std::move(a);
				return 1;
			}
			if (hw->eof && hw->ready.empty() && !sw->ready.empty()) { drop_hw("it gave fewer frames than software"); continue; }
			if (sw->eof && sw->ready.empty() && !hw->ready.empty()) { drop_hw("it gave more frames than software"); continue; }
			if (sw->eof && hw->eof) { // the whole stream was compared
				if (compared > 0) drop_sw(true);
				else hw.reset();
				continue;
			}
			return 0;
		}
	}
};

} // namespace

std::unique_ptr<Decoder> make_decoder() { return std::make_unique<FfDecoder>(); }
