#include "vmaf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "decode.h"

extern "C" {
#include <libvmaf/libvmaf.h>
#include <libvmaf/model.h>
#include <libvmaf/picture.h>
}

namespace {

// Scores picture pairs with libvmaf in-process; every score stays in memory.
class Scorer {
	VmafContext* ctx = nullptr;
	VmafModel* model = nullptr;
	unsigned count = 0;

	static bool to_picture(const Picture& p, VmafPicture& pic) {
		if (!p.fmt.planar || p.fmt.rgb) { fprintf(stderr, "vmaf: unsupported pixel format\n"); return false; }
		enum VmafPixelFormat fmt = p.fmt.chroma == 420 ? VMAF_PIX_FMT_YUV420P : p.fmt.chroma == 422 ? VMAF_PIX_FMT_YUV422P : VMAF_PIX_FMT_YUV444P;
		if (vmaf_picture_alloc(&pic, fmt, p.fmt.depth, p.width, p.height) < 0) return false;
		for (int i = 0; i < 3; i++) {
			size_t row = (size_t)pic.w[i] * (p.fmt.depth > 8 ? 2 : 1);
			for (unsigned y = 0; y < pic.h[i]; y++) memcpy((uint8_t*)pic.data[i] + y * pic.stride[i], p.data[i] + (size_t)y * p.stride[i], row);
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

	bool add(const Picture& ref, const Picture& dis) {
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

bool vmaf_score(const RefInfo& ref, const FrameFeed& next_ref, Codec codec, const PacketFeed& next_packet, const std::vector<uint8_t>& extradata, const Shape& enc,
                const ShownFeed& shown, Fmt score_fmt, const VmafOptions& opt, const ScaleOptions& scale, std::vector<double>& scores, RunProgress* rp) {
	if (rp) rp->begin(2, std::max<long long>(1, ref.frames));
	auto decoder = make_decoder();
	if (!decoder->open(codec, extradata)) return false;

	const bool convert_dis = enc.w != ref.width || enc.h != ref.height || enc.pix != score_fmt; // scale back, change format

	Scorer scorer;
	auto up = make_converter();
	Picture d, u, r;
	long long m = 0, mo = 0, k = 0; // decoded frames in, converted frames out, original frame index
	bool ok = scorer.open(opt);

	// Hold `shown` (decoded frame number mi, at the original resolution) for every original frame up to the next one.
	auto show = [&](const Picture& pic, int64_t mi) {
		uint32_t on_screen;
		while (ok && shown((size_t)k, on_screen) && on_screen == mi) {
			int rr = next_ref(r);
			if (rr < 0) { ok = false; break; }
			if (rr == 0) break;
			ok = scorer.add(r, pic);
			k++;
			if (rp) rp->step();
		}
	};
	// libplacebo holds frames back until it has seen the next one, so converted frames are collected as they appear.
	auto collect_converted = [&] {
		int got;
		while (ok && (got = up->pop(u)) == 1) show(u, mo++);
		ok = ok && got >= 0;
	};
	auto pump = [&] {
		int got;
		while (ok && (got = decoder->receive(d)) == 1) {
			if (d.width != enc.w || d.height != enc.h || d.fmt != enc.pix) {
				fprintf(stderr, "decoded stream is %dx%d %d-bit %d, expected %dx%d %d-bit %d\n", d.width, d.height, d.fmt.depth, d.fmt.chroma, enc.w, enc.h, enc.pix.depth, enc.pix.chroma);
				ok = false; break;
			}
			// Encoders do not necessarily write the colour description, so restore the source's before converting.
			d.color = ref.color;
			if (!convert_dis) show(d, m);
			else {
				d.pts = m;
				d.duration = 1; // libplacebo needs it to know when the last frame ends; the packets carry none
				if (m == 0 && !up->open(d, enc.fps.inverse(), ref.width, ref.height, score_fmt, scale)) { ok = false; break; }
				if (!up->push(&d)) { fprintf(stderr, "scaling the decoded frame back failed\n"); ok = false; break; }
				collect_converted();
			}
			m++;
		}
		ok = ok && got >= 0;
	};

	for (std::vector<uint8_t> packet; ok && next_packet(packet);) {
		ok = decoder->send(packet.data(), packet.size());
		if (ok) pump();
	}
	if (ok) { decoder->flush(); pump(); }
	if (ok && convert_dis) { ok = up->push(nullptr); collect_converted(); }
	if (ok && k == 0) { fprintf(stderr, "no frames were decoded\n"); ok = false; }
	uint32_t extra;
	if (ok && (shown((size_t)k, extra) || next_ref(r) == 1)) { fprintf(stderr, "encoded stream does not cover the reference frames\n"); ok = false; }
	return ok && scorer.finish(scores);
}
