// Sweep encoder option combinations: for each one, encode in-process, decode back and score every
// frame with VMAF, then report which configurations are the best in at least 95% of frames.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <sstream>
#include <string>

#include "encoder.h"
#include "source.h"
#include "vmaf.h"

static constexpr double BEST_FRAME_SHARE = 0.95;

struct Run {
	int64_t bitrate;        // target, kbps
	std::string label;      // option set
	double kbps = 0;        // actual
	std::vector<double> scores; // per frame, display order
	double penalty = 0;     // VMAF points deducted for exceeding the target bitrate
	double best_share = 0;  // share of frames where this run is at least as good as every other run of its bitrate group
};

// Running under a CBR target is the encoder's own loss, so only overshoot is penalized.
// Default slope: the 3000 -> 6000 kbps sweep gains roughly 2.5 VMAF over +100%, i.e. ~0.025 per percent.
static double over_penalty = 0.025; // VMAF points per percent over target

static std::vector<std::string> split(const std::string& s) {
	std::vector<std::string> out;
	std::stringstream ss(s);
	for (std::string t; std::getline(ss, t, ',');) out.push_back(t);
	return out;
}

static bool codec_id(const std::string& encoder, const std::string& codec, AVCodecID& id) {
	if (encoder == "nvenc") {
		id = codec == "h264" ? AV_CODEC_ID_H264 : codec == "hevc" ? AV_CODEC_ID_HEVC : codec == "av1" ? AV_CODEC_ID_AV1 : AV_CODEC_ID_NONE;
	} else {
		const AVCodec* c = avcodec_find_encoder_by_name(encoder.c_str());
		id = c ? c->id : AV_CODEC_ID_NONE;
	}
	return id != AV_CODEC_ID_NONE;
}

static bool run_one(const std::string& input, const std::string& encoder, AVCodecID id, Params p, const VmafOptions& vo, Run& out) {
	Source src;
	if (!src.open(input)) return false;
	p.width = src.width; p.height = src.height;
	p.framerate = src.framerate;
	p.gop = (int)(av_q2d(p.framerate) * 2);

	std::vector<std::vector<uint8_t>> packets;
	auto enc = make_encoder(encoder);
	// Packets arrive in coding order, which is also the order the decoder needs them.
	if (!enc->open(p, [&](const uint8_t* d, size_t n, int64_t, bool) { packets.emplace_back(d, d + n); })) return false;

	AVFrame* frame = av_frame_alloc();
	long long frames = 0;
	bool ok = true;
	int r;
	while (ok && (r = src.next(frame)) == 1) { ok = enc->encode(frame); av_frame_unref(frame); frames++; }
	ok = ok && r >= 0 && enc->encode(nullptr);
	av_frame_free(&frame);
	if (!ok) return false;
	out.kbps = enc->stats.bytes * 8 / (frames / av_q2d(p.framerate)) / 1000;
	enc.reset();

	Source ref;
	if (!ref.open(input) || !vmaf_score(ref, id, packets, vo, out.scores)) return false;
	if ((long long)out.scores.size() != frames) { fprintf(stderr, "%zu scores for %lld frames\n", out.scores.size(), frames); return false; }
	return true;
}

int main(int argc, char** argv) {
	if (argc < 5) {
		fprintf(stderr, "usage: ves <input> <encoder> <codec> <bitrate-kbps[,..]> [key=value[,value..] ...]\n"
		                "  encoder: nvenc | any libavcodec encoder name (codec is ignored for the latter)\n"
		                "  comma separated values are swept; every combination is tested\n"
		                "  vmaf.model=<spec> and vmaf.threads=<n> configure scoring, over.penalty=<vmaf points per %% over target>,\n"
		                "  other keys go to the encoder\n");
		return 1;
	}
	std::string input = argv[1], encoder = argv[2];
	Params base;
	VmafOptions vo;
	base.codec = argv[3];
	std::vector<std::string> bitrates = split(argv[4]);

	std::vector<Options> combos(1);
	for (int i = 5; i < argc; i++) {
		std::string a = argv[i];
		size_t eq = a.find('=');
		if (eq == std::string::npos) { fprintf(stderr, "bad option '%s', expected key=value\n", argv[i]); return 1; }
		std::string k = a.substr(0, eq), v = a.substr(eq + 1);
		if (k == "vmaf.model") { vo.model = v; continue; }
		if (k == "vmaf.threads") { vo.threads = atoi(v.c_str()); continue; }
		if (k == "over.penalty") { over_penalty = atof(v.c_str()); continue; }
		std::vector<Options> next;
		for (auto& c : combos) for (auto& val : split(v)) { Options n = c; n[k] = val; next.push_back(n); }
		combos = std::move(next);
	}
	AVCodecID id;
	if (!codec_id(encoder, base.codec, id)) { fprintf(stderr, "unknown encoder/codec '%s'/'%s'\n", encoder.c_str(), base.codec.c_str()); return 1; }

	std::vector<Run> runs;
	int failed = 0;
	for (auto& br : bitrates) {
		for (auto& c : combos) {
			Run r;
			r.bitrate = atoll(br.c_str());
			for (auto& [k, v] : c) r.label += (r.label.empty() ? "" : " ") + k + "=" + v;
			Params p = base;
			p.bitrate = r.bitrate * 1000;
			p.options = c;
			if (!run_one(input, encoder, id, p, vo, r)) { fprintf(stderr, "[%lld kbps] %s: FAILED\n", (long long)r.bitrate, r.label.c_str()); failed++; continue; }
			r.penalty = std::max(0.0, r.kbps / r.bitrate - 1) * 100 * over_penalty;
			std::vector<double> s = r.scores;
			std::sort(s.begin(), s.end());
			printf("[%lld kbps] %s: %.0f kbps, vmaf mean %.3f min %.3f p5 %.3f\n", (long long)r.bitrate, r.label.c_str(), r.kbps,
			       std::accumulate(s.begin(), s.end(), 0.0) / s.size(), s.front(), s[(size_t)(s.size() * 0.05)]);
			fflush(stdout);
			runs.push_back(std::move(r));
		}
	}

	// Within each bitrate group, count the frames on which a run is at least as good as all others.
	std::stable_sort(runs.begin(), runs.end(), [](const Run& a, const Run& b) { return a.bitrate > b.bitrate; });
	for (size_t g = 0; g < runs.size();) {
		size_t e = g;
		while (e < runs.size() && runs[e].bitrate == runs[g].bitrate) e++;
		size_t frames = runs[g].scores.size();
		for (size_t i = g; i < e; i++) {
			size_t best = 0;
			for (size_t f = 0; f < frames; f++) {
				double top = 0;
				for (size_t j = g; j < e; j++) top = std::max(top, runs[j].scores[f] - runs[j].penalty);
				best += runs[i].scores[f] - runs[i].penalty >= top - 1e-6;
			}
			runs[i].best_share = (double)best / frames;
		}
		std::sort(runs.begin() + g, runs.begin() + e, [](const Run& a, const Run& b) { return a.best_share > b.best_share; });
		printf("\n== %lld kbps: best in >= %.0f%% of frames ==\n", (long long)runs[g].bitrate, BEST_FRAME_SHARE * 100);
		bool any = false;
		for (size_t i = g; i < e; i++) {
			bool win = runs[i].best_share >= BEST_FRAME_SHARE;
			any |= win;
			printf("%s %5.1f%%  %s  (%.0f kbps, penalty %.2f)\n", win ? "*" : " ", runs[i].best_share * 100, runs[i].label.c_str(), runs[i].kbps, runs[i].penalty);
		}
		if (!any) printf("  no configuration reaches %.0f%%; the top entry is the closest\n", BEST_FRAME_SHARE * 100);
		g = e;
	}
	return failed ? 2 : 0;
}
