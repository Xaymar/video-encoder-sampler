#include "vmaf.h"

#include <cstdio>
#include <cstring>

extern "C" {
#include <libavutil/pixdesc.h>
#include <libvmaf/libvmaf.h>
#include <libvmaf/model.h>
#include <libvmaf/picture.h>
}

namespace {

// Scores frame pairs with libvmaf in-process; every score stays in memory.
class Scorer {
	VmafContext* ctx = nullptr;
	VmafModel* model = nullptr;
	unsigned count = 0;

	static bool to_picture(const AVFrame* f, VmafPicture& pic) {
		const AVPixFmtDescriptor* d = av_pix_fmt_desc_get((AVPixelFormat)f->format);
		if (!d || !(d->flags & AV_PIX_FMT_FLAG_PLANAR) || (d->flags & AV_PIX_FMT_FLAG_RGB) || d->nb_components < 3 || d->comp[0].depth > 16) {
			fprintf(stderr, "vmaf: unsupported pixel format %d\n", f->format);
			return false;
		}
		enum VmafPixelFormat fmt = d->log2_chroma_w && d->log2_chroma_h ? VMAF_PIX_FMT_YUV420P : d->log2_chroma_w ? VMAF_PIX_FMT_YUV422P : VMAF_PIX_FMT_YUV444P;
		if (vmaf_picture_alloc(&pic, fmt, d->comp[0].depth, f->width, f->height) < 0) return false;
		for (int p = 0; p < 3; p++) {
			size_t row = (size_t)pic.w[p] * (d->comp[0].depth > 8 ? 2 : 1);
			for (unsigned y = 0; y < pic.h[p]; y++) memcpy((uint8_t*)pic.data[p] + y * pic.stride[p], f->data[p] + (size_t)y * f->linesize[p], row);
		}
		return true;
	}

public:
	~Scorer() {
		if (ctx) vmaf_close(ctx);
		if (model) vmaf_model_destroy(model);
	}

	bool open(const VmafOptions& o) {
		VmafConfiguration cfg{};
		cfg.log_level = VMAF_LOG_LEVEL_WARNING;
		cfg.n_threads = o.threads;
		if (vmaf_init(&ctx, cfg) < 0) { fprintf(stderr, "vmaf: init failed\n"); return false; }
		VmafModelConfig mc{"vmaf", 0};
		bool from_path = o.model.rfind("path=", 0) == 0;
		std::string spec = o.model.substr(o.model.find('=') == std::string::npos ? 0 : o.model.find('=') + 1);
		int r = from_path ? vmaf_model_load_from_path(&model, &mc, spec.c_str()) : vmaf_model_load(&model, &mc, spec.c_str());
		if (r < 0) { fprintf(stderr, "vmaf: cannot load model '%s'\n", o.model.c_str()); return false; }
		if (vmaf_use_features_from_model(ctx, model) < 0) { fprintf(stderr, "vmaf: model features unavailable\n"); return false; }
		return true;
	}

	bool add(const AVFrame* ref, const AVFrame* dis) {
		VmafPicture rp{}, dp{};
		if (!to_picture(ref, rp)) return false;
		if (!to_picture(dis, dp)) { vmaf_picture_unref(&rp); return false; }
		if (vmaf_read_pictures(ctx, &rp, &dp, count) < 0) { fprintf(stderr, "vmaf: scoring frame %u failed\n", count); return false; }
		count++;
		return true;
	}

	bool finish(std::vector<double>& scores) {
		if (vmaf_read_pictures(ctx, nullptr, nullptr, 0) < 0) { fprintf(stderr, "vmaf: flushing failed\n"); return false; }
		scores.assign(count, 0);
		for (unsigned i = 0; i < count; i++) {
			if (vmaf_score_at_index(ctx, model, &scores[i], i) < 0) { fprintf(stderr, "vmaf: no score for frame %u\n", i); return false; }
		}
		return true;
	}
};

} // namespace

bool vmaf_score(Source& ref, AVCodecID codec, const std::vector<std::vector<uint8_t>>& packets, const Shape& enc,
                const std::vector<uint32_t>& shown_map, AVPixelFormat score_fmt, const VmafOptions& opt, const ScaleOptions& scale, std::vector<double>& scores) {
	const AVCodec* c = avcodec_find_decoder(codec);
	AVCodecContext* dec = c ? avcodec_alloc_context3(c) : nullptr;
	// Without VUI reorder info the H.264 decoder guesses the B-frame depth and can drop or misorder early frames;
	// strict mode makes it assume the maximum, so every frame comes out in display order.
	if (dec) dec->strict_std_compliance = FF_COMPLIANCE_STRICT;
	if (!dec || avcodec_open2(dec, c, nullptr) < 0) { fprintf(stderr, "no decoder for encoded stream\n"); avcodec_free_context(&dec); return false; }

	const bool convert_dis = enc.w != ref.width || enc.h != ref.height || enc.pix != score_fmt; // scale back, change format
	const bool convert_ref = ref.pix != score_fmt;

	bool ok = true;
	{
		Scorer scorer;
		Scaler up, ref_conv;
		AVFrame* d = av_frame_alloc();
		AVFrame* u = av_frame_alloc();
		AVFrame* r = av_frame_alloc();
		AVFrame* t = av_frame_alloc();
		AVPacket* pkt = av_packet_alloc();
		long long m = 0, mo = 0, k = 0; // decoded frames in, converted frames out, original frame index
		long long ref_in = 0;
		bool ref_done = false;

		ok = scorer.open(opt);

		// Next reference frame in the scoring format. The converter holds frames back, so feed it until one comes out.
		auto next_ref = [&](AVFrame* out) -> int {
			if (!convert_ref) return ref.next(out);
			for (;;) {
				int got = ref_conv.pop(out);
				if (got != 0) return got;
				if (ref_done) return 0;
				int rr = ref.next(t);
				if (rr < 0) return rr;
				if (rr == 0) { ref_done = true; if (!ref_conv.push(nullptr)) return -1; continue; }
				t->duration = 1;
				if (ref_in == 0 && !ref_conv.open(t, av_inv_q(ref.framerate), ref.width, ref.height, score_fmt, scale)) return -1;
				ref_in++;
				if (!ref_conv.push(t)) return -1;
				av_frame_unref(t);
			}
		};

		// Hold `shown` (decoded frame number mi, at the original resolution) for every original frame up to the next one.
		auto show = [&](const AVFrame* shown, int64_t mi) {
			while (ok && k < (long long)shown_map.size() && shown_map[k] == mi) {
				int rr = next_ref(r);
				if (rr < 0) { ok = false; break; }
				if (rr == 0) break;
				ok = scorer.add(r, shown);
				av_frame_unref(r);
				k++;
			}
		};
		// libplacebo holds frames back until it has seen the next one, so converted frames are collected as they appear.
		auto collect_converted = [&] {
			int got;
			while (ok && (got = up.pop(u)) == 1) { show(u, mo++); av_frame_unref(u); }
			ok = ok && got >= 0;
		};
		auto pump = [&] {
			while (ok && avcodec_receive_frame(dec, d) >= 0) {
				if (d->width != enc.w || d->height != enc.h || d->format != enc.pix) {
					fprintf(stderr, "decoded stream is %dx%d fmt %d, expected %dx%d fmt %d\n", d->width, d->height, d->format, enc.w, enc.h, enc.pix);
					ok = false; break;
				}
				// Encoders do not necessarily write the colour description, so restore the source's before converting.
				d->color_range = ref.range; d->colorspace = ref.space; d->color_primaries = ref.primaries; d->color_trc = ref.trc;
				if (!convert_dis) show(d, m);
				else {
					d->pts = m;
					d->duration = 1; // libplacebo needs it to know when the last frame ends; the packets carry none
					if (m == 0 && !up.open(d, av_inv_q(enc.fps), ref.width, ref.height, score_fmt, scale)) { ok = false; break; }
					if (!up.push(d)) { fprintf(stderr, "scaling the decoded frame back failed\n"); ok = false; break; }
					collect_converted();
				}
				av_frame_unref(d);
				m++;
			}
		};

		for (size_t i = 0; ok && i < packets.size(); i++) {
			av_new_packet(pkt, (int)packets[i].size());
			memcpy(pkt->data, packets[i].data(), packets[i].size());
			ok = avcodec_send_packet(dec, pkt) >= 0;
			av_packet_unref(pkt);
			if (ok) pump();
		}
		if (ok) { avcodec_send_packet(dec, nullptr); pump(); }
		if (ok && convert_dis) { ok = up.push(nullptr); collect_converted(); }
		if (ok && k == 0) { fprintf(stderr, "no frames were decoded\n"); ok = false; }
		if (ok && (k != (long long)shown_map.size() || next_ref(r) == 1)) { fprintf(stderr, "encoded stream does not cover the reference frames\n"); ok = false; }
		if (ok) ok = scorer.finish(scores);

		av_frame_free(&d); av_frame_free(&u); av_frame_free(&r); av_frame_free(&t); av_packet_free(&pkt);
	}

	avcodec_free_context(&dec);
	return ok;
}
