#include "vmaf.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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

	bool build(const Source& s, const VmafOptions& o, const std::string& log) {
		char args[256];
		snprintf(args, sizeof args, "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=1/1",
		         s.width, s.height, (int)AV_PIX_FMT_YUV420P, s.framerate.den, s.framerate.num);
		const AVFilter* buf = avfilter_get_by_name("buffer");
		const AVFilter* lv = avfilter_get_by_name("libvmaf");
		if (!lv) { fprintf(stderr, "this FFmpeg build has no libvmaf filter\n"); return false; }
		vmaf = avfilter_graph_alloc_filter(g, lv, "vmaf");
		av_opt_set(vmaf, "log_fmt", "json", AV_OPT_SEARCH_CHILDREN);
		av_opt_set(vmaf, "log_path", log.c_str(), AV_OPT_SEARCH_CHILDREN);
		av_opt_set(vmaf, "model", o.model.c_str(), AV_OPT_SEARCH_CHILDREN);
		av_opt_set_int(vmaf, "n_threads", o.threads, AV_OPT_SEARCH_CHILDREN);
		if (avfilter_init_str(vmaf, nullptr) < 0) { fprintf(stderr, "libvmaf init failed (model '%s'?)\n", o.model.c_str()); return false; }
		return avfilter_graph_create_filter(&dis, buf, "dis", args, nullptr, g) >= 0
			&& avfilter_graph_create_filter(&ref, buf, "ref", args, nullptr, g) >= 0
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

bool vmaf_score(Source& ref, AVCodecID codec, const std::vector<std::vector<uint8_t>>& packets,
                const VmafOptions& opt, std::vector<double>& scores) {
	static std::atomic<int> counter;
	std::string log = (std::filesystem::temp_directory_path() / ("ves_vmaf_" + std::to_string(counter++) + ".json")).string();

	const AVCodec* c = avcodec_find_decoder(codec);
	AVCodecContext* dec = c ? avcodec_alloc_context3(c) : nullptr;
	// Without VUI reorder info the H.264 decoder guesses the B-frame depth and can drop or misorder early frames;
	// strict mode makes it assume the maximum, so every frame comes out in display order.
	if (dec) dec->strict_std_compliance = FF_COMPLIANCE_STRICT;
	if (!dec || avcodec_open2(dec, c, nullptr) < 0) { fprintf(stderr, "no decoder for encoded stream\n"); avcodec_free_context(&dec); return false; }

	bool ok = true;
	{
		Graph gr;
		AVFrame* d = av_frame_alloc();
		AVFrame* r = av_frame_alloc();
		AVFrame* o = av_frame_alloc();
		AVPacket* pkt = av_packet_alloc();
		long long n = 0;

		auto drain_sink = [&] { while (av_buffersink_get_frame(gr.sink, o) >= 0) av_frame_unref(o); };
		auto pump = [&] { // pair every decoded frame with its reference frame
			while (ok && avcodec_receive_frame(dec, d) >= 0) {
				if (d->width != ref.width || d->height != ref.height || d->format != AV_PIX_FMT_YUV420P) {
					fprintf(stderr, "decoded stream is %dx%d fmt %d, expected %dx%d yuv420p\n", d->width, d->height, d->format, ref.width, ref.height);
					ok = false; break;
				}
				if (ref.next(r) != 1) { fprintf(stderr, "reference ended before encoded stream\n"); ok = false; break; }
				d->pts = r->pts = n++;
				ok = av_buffersrc_add_frame(gr.dis, d) >= 0 && av_buffersrc_add_frame(gr.ref, r) >= 0;
				av_frame_unref(d); av_frame_unref(r);
				drain_sink();
			}
		};

		ok = gr.build(ref, opt, log);
		for (size_t i = 0; ok && i < packets.size(); i++) {
			av_new_packet(pkt, (int)packets[i].size());
			memcpy(pkt->data, packets[i].data(), packets[i].size());
			ok = avcodec_send_packet(dec, pkt) >= 0;
			av_packet_unref(pkt);
			if (ok) pump();
		}
		if (ok) { avcodec_send_packet(dec, nullptr); pump(); }
		if (ok) { ok = av_buffersrc_add_frame(gr.dis, nullptr) >= 0 && av_buffersrc_add_frame(gr.ref, nullptr) >= 0; drain_sink(); }
		if (ok && ref.next(r) == 1) { fprintf(stderr, "encoded stream has fewer frames than the reference\n"); ok = false; }

		av_frame_free(&d); av_frame_free(&r); av_frame_free(&o); av_packet_free(&pkt);
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
