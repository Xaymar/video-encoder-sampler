// Decoder on top of libavcodec, for the encoded streams that are scored.
//
// The software decoder is the truth. With hardware decoding on (--hw-decode, on the GPU --decode-gpu picks), a stream is decoded by the GPU as well,
// and both run side by side on its first frames: as long as they are compared, the software decode's frames are the ones handed out, and once a
// frame differs (or the GPU cannot decode the stream, gives fewer frames, a format we cannot use, ...) the hardware decoder is dropped and software
// carries on from there, so the scores are those of a software decode either way. After the check the software decoder is dropped instead.
// The first stream of a kind (codec, bit depth, chroma, size, device) in a process is checked all the way through, later ones on their first
// frames (VES_HW_DECODE_VERIFY=<frames> or "all"). A device that once differed for a kind is not used for it again.
// Only FFmpeg's own decoders on a hardware device (hwaccel) are used, not the dedicated ones (h264_cuvid, ...): they parse, reorder and crop
// the stream with the same code as the software decoder, and only the reconstruction runs on the GPU. (AV1 is the exception: software is libdav1d,
// the GPU gets FFmpeg's own AV1 decoder, which has no software path of its own; the check covers that as well.)
#include "decode.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
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

constexpr long long VERIFY_FRAMES = 30; // checked on each later stream of a kind

long long verify_frames() {
	if (const char* e = getenv("VES_HW_DECODE_VERIFY")) return strcmp(e, "all") == 0 ? -1 : atoll(e);
	return VERIFY_FRAMES;
}

std::mutex g_mutex;
std::set<std::string> g_bad;          // kind|type|device that differed from software or could not decode
std::set<std::string> g_checked_all;  // kinds whose first stream is (being) checked all the way
std::set<std::string> g_told;         // messages said once

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
	std::string kind, hw_key, hw_name; // the stream's kind, kind|type|device of the hardware decoder, and what to call it
	long long check = 0;               // frames still to compare; -1: all of them
	long long compared = 0;
	bool full_check = false;

	void drop_hw(const std::string& why) {
		{
			std::lock_guard lock(g_mutex);
			g_bad.insert(hw_key);
		}
		tell_once("drop|" + hw_key, "scored streams: hardware decoding (" + hw_name + ") is not used for " + kind + ": " + why + "; decoding in software");
		hw.reset();
	}

	void drop_sw() {
		tell_once("ok|" + hw_key + (full_check ? "|all" : ""), "scored streams: hardware decoding (" + hw_name + ") matched software on " +
		          (full_check ? "all " + std::to_string(compared) + " frames of the first " + kind + " stream" : "the first " + std::to_string(compared) + " frames of a " + kind + " stream"));
		sw.reset();
	}

public:
	~FfDecoder() override { av_packet_free(&pkt); }

	bool open(Codec codec, const std::vector<uint8_t>& extradata, const Shape* enc) override {
		const AVCodecID id = codec == Codec::H264 ? AV_CODEC_ID_H264 : codec == Codec::HEVC ? AV_CODEC_ID_HEVC : codec == Codec::AV1 ? AV_CODEC_ID_AV1 : AV_CODEC_ID_NONE;
		sw = std::make_unique<Dec>();
		if (id == AV_CODEC_ID_NONE || !sw->open(id, extradata, "", "")) { fprintf(stderr, "no decoder for encoded stream\n"); return false; }
		if (!enc) return true; // no kind to keep verdicts for: software only
		kind = std::string(avcodec_get_name(id)) + " " + std::to_string(enc->w) + "x" + std::to_string(enc->h) + " " + std::to_string(enc->pix.depth) + "-bit " +
		       (enc->pix.chroma == 420 ? "4:2:0" : enc->pix.chroma == 422 ? "4:2:2" : "4:4:4");
		// The first hardware decoder that opens and is not known to be bad for this kind of stream.
		std::vector<HwChoice> choices = hw_decode_choices(id);
		if (getenv("VES_HW_DECODE_TEST")) choices = {{"test", ""}};
		for (const HwChoice& c : choices) {
			const std::string key = kind + "|" + c.type + "|" + c.device;
			{
				std::lock_guard lock(g_mutex);
				if (g_bad.count(key)) continue;
			}
			auto d = std::make_unique<Dec>();
			hw_key = key;
			hw_name = c.type + (c.device.empty() ? "" : ", device " + c.device);
			if (!d->open(id, extradata, c.type, c.device)) { // no such device here, or it cannot decode the codec
				std::lock_guard lock(g_mutex);
				g_bad.insert(key);
				continue;
			}
			hw = std::move(d);
			break;
		}
		if (!hw) {
			if (!choices.empty()) tell_once("none|" + kind, "scored streams: no hardware decoder can be used for " + kind + " streams: decoding in software");
			return true;
		}
		std::lock_guard lock(g_mutex);
		check = verify_frames();
		if (check == 0) check = 1; // the first frame at least: it tells whether the GPU decodes the stream at all
		if (g_checked_all.insert(hw_key).second) { full_check = true; check = -1; }
		return true;
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
				} else if (++compared == check) drop_sw();
				out = std::move(a);
				return 1;
			}
			if (hw->eof && hw->ready.empty() && !sw->ready.empty()) { drop_hw("it gave fewer frames than software"); continue; }
			if (sw->eof && sw->ready.empty() && !hw->ready.empty()) { drop_hw("it gave more frames than software"); continue; }
			if (sw->eof && hw->eof) { // the whole stream was compared
				if (full_check && compared > 0) drop_sw();
				else hw.reset();
				continue;
			}
			return 0;
		}
	}
};

} // namespace

std::unique_ptr<Decoder> make_decoder() { return std::make_unique<FfDecoder>(); }
