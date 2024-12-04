#include "vmaf.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <sstream>

extern "C" {
#include <libavfilter/avfilter.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include <libavutil/opt.h>
}

namespace {

struct Graph {
	AVFilterGraph* g = avfilter_graph_alloc();
	AVFilterContext *dis = nullptr, *ref = nullptr, *vmaf = nullptr, *sink = nullptr;
	~Graph() { avfilter_graph_free(&g); }

	bool build(const AVFrame* dis_t, const AVFrame* ref_t, AVRational tb, const VmafOptions& o, const std::string& log) {
		const AVFilter* buf = avfilter_get_by_name("buffer");
		const AVFilter* lv = avfilter_get_by_name("libvmaf");
		if (!lv) { fprintf(stderr, "this FFmpeg build has no libvmaf filter\n"); return false; }
		vmaf = avfilter_graph_alloc_filter(g, lv, "vmaf");
		av_opt_set(vmaf, "log_fmt", "json", AV_OPT_SEARCH_CHILDREN);
		av_opt_set(vmaf, "log_path", log.c_str(), AV_OPT_SEARCH_CHILDREN);
		av_opt_set(vmaf, "model", o.model.c_str(), AV_OPT_SEARCH_CHILDREN);
		av_opt_set_int(vmaf, "n_threads", o.threads, AV_OPT_SEARCH_CHILDREN);
		if (avfilter_init_str(vmaf, nullptr) < 0) { fprintf(stderr, "libvmaf init failed (model '%s'?)\n", o.model.c_str()); return false; }
		return avfilter_graph_create_filter(&dis, buf, "dis", buffer_args(dis_t, tb).c_str(), nullptr, g) >= 0
			&& avfilter_graph_create_filter(&ref, buf, "ref", buffer_args(ref_t, tb).c_str(), nullptr, g) >= 0
			&& avfilter_graph_create_filter(&sink, avfilter_get_by_name("buffersink"), "out", nullptr, nullptr, g) >= 0
			&& avfilter_link(dis, 0, vmaf, 0) >= 0 && avfilter_link(ref, 0, vmaf, 1) >= 0 && avfilter_link(vmaf, 0, sink, 0) >= 0
			&& avfilter_graph_config(g, nullptr) >= 0;
	}
};

void parse_scores(const std::string& text, std::vector<double>& out) {
	// Each frame object holds "frameNum" followed by its metrics; the first "vmaf" after it is that frame's score.
	for (size_t pos = text.find("\"frameNum\""); pos != std::string::npos; pos = text.find("\"frameNum\"", pos + 1)) {
		size_t m = text.find("\"vmaf\"", pos);
		if (m == std::string::npos) break;
		out.push_back(strtod(text.c_str() + text.find(':', m) + 1, nullptr));
	}
}

} // namespace

bool vmaf_score(Source& ref, AVCodecID codec, const std::vector<std::vector<uint8_t>>& packets, const Shape& enc,
                const VmafOptions& opt, const ScaleOptions& scale, std::vector<double>& scores) {
	static std::atomic<int> counter;
	std::string log = (std::filesystem::temp_directory_path() / ("ves_vmaf_" + std::to_string(counter++) + ".json")).string();

	const AVCodec* c = avcodec_find_decoder(codec);
	AVCodecContext* dec = c ? avcodec_alloc_context3(c) : nullptr;
	// Without VUI reorder info the H.264 decoder guesses the B-frame depth and can drop or misorder early frames;
	// strict mode makes it assume the maximum, so every frame comes out in display order.
	if (dec) dec->strict_std_compliance = FF_COMPLIANCE_STRICT;
	if (!dec || avcodec_open2(dec, c, nullptr) < 0) { fprintf(stderr, "no decoder for encoded stream\n"); avcodec_free_context(&dec); return false; }

	// Original frame k shows encoded frame floor(k * enc.fps / ref.fps): dropped frames are held, never interpolated.
	const int64_t num = (int64_t)enc.fps.num * ref.framerate.den, den = (int64_t)enc.fps.den * ref.framerate.num;
	const bool rescale = enc.w != ref.width || enc.h != ref.height;
	const AVRational tb = av_inv_q(ref.framerate);

	bool ok = true;
	{
		Graph gr;
		Scaler up;
		AVFrame* d = av_frame_alloc();
		AVFrame* u = av_frame_alloc();
		AVFrame* r = av_frame_alloc();
		AVFrame* o = av_frame_alloc();
		AVPacket* pkt = av_packet_alloc();
		long long m = 0, mo = 0, k = 0; // decoded frames in, scaled frames out, original frame index
		bool graph_ready = false;

		auto drain_sink = [&] { while (av_buffersink_get_frame(gr.sink, o) >= 0) av_frame_unref(o); };
		// Hold `shown` (decoded frame number mi, at the original resolution) for every original frame up to the next one.
		auto show = [&](const AVFrame* shown, int64_t mi) {
			while (ok && k * num / den == mi) {
				int rr = ref.next(r);
				if (rr < 0) { ok = false; break; }
				if (rr == 0) break;
				AVFrame* dup = av_frame_clone(shown);
				dup->pts = k;
				if (!graph_ready) {
					ok = gr.build(dup, r, tb, opt, log);
					graph_ready = ok;
				}
				ok = ok && av_buffersrc_add_frame(gr.dis, dup) >= 0 && av_buffersrc_add_frame(gr.ref, r) >= 0;
				av_frame_free(&dup);
				av_frame_unref(r);
				drain_sink();
				k++;
			}
		};
		// libplacebo holds frames back until it has seen the next one, so scaled frames are collected as they appear.
		auto collect_scaled = [&] {
			int got;
			while (ok && (got = up.pop(u)) == 1) { show(u, mo++); av_frame_unref(u); }
			ok = ok && got >= 0;
		};
		auto pump = [&] {
			while (ok && avcodec_receive_frame(dec, d) >= 0) {
				if (d->width != enc.w || d->height != enc.h || d->format != AV_PIX_FMT_YUV420P) {
					fprintf(stderr, "decoded stream is %dx%d fmt %d, expected %dx%d yuv420p\n", d->width, d->height, d->format, enc.w, enc.h);
					ok = false; break;
				}
				// Encoders do not necessarily write the colour description, so restore the source's before scaling.
				d->color_range = ref.range; d->colorspace = ref.space; d->color_primaries = ref.primaries; d->color_trc = ref.trc;
				if (!rescale) show(d, m);
				else {
					d->pts = m;
					d->duration = 1; // libplacebo needs it to know when the last frame ends; the packets carry none
					if (m == 0 && !up.open(d, av_inv_q(enc.fps), ref.width, ref.height, scale)) { ok = false; break; }
					if (!up.push(d)) { fprintf(stderr, "scaling the decoded frame back failed\n"); ok = false; break; }
					collect_scaled();
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
		if (ok && rescale) { ok = up.push(nullptr); collect_scaled(); }
		if (ok && !graph_ready) { fprintf(stderr, "no frames were decoded\n"); ok = false; }
		if (ok) { ok = av_buffersrc_add_frame(gr.dis, nullptr) >= 0 && av_buffersrc_add_frame(gr.ref, nullptr) >= 0; drain_sink(); }
		if (ok && ref.next(r) == 1) { fprintf(stderr, "encoded stream has fewer frames than the reference\n"); ok = false; }

		av_frame_free(&d); av_frame_free(&u); av_frame_free(&r); av_frame_free(&o); av_packet_free(&pkt);
	} // graph freed here: libvmaf writes its log on teardown

	avcodec_free_context(&dec);
	if (ok) {
		std::ifstream f(log);
		std::stringstream ss; ss << f.rdbuf();
		scores.clear();
		parse_scores(ss.str(), scores);
		if (scores.empty()) { fprintf(stderr, "no scores found in '%s'\n", log.c_str()); return false; }
	}
	std::error_code ec; std::filesystem::remove(log, ec);
	return ok;
}
