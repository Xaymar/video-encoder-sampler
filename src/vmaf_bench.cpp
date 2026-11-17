// ves-vmaf-bench: times the GPU VMAF scorer (vk_vmaf.cpp) alone, on raw frames held in memory, so it can be compared with other GPU VMAF engines
// (libvmaf-fast's bench_vmaf_v1.py, libvmaf on the CPU) on the same frames. Not part of ves itself.
//
//   ves-vmaf-bench --ref REF.yuv --dis DIS.yuv --size WxH [--bits 10] [--chroma 420] --model M.json [--model ...]
//                  [--frames 24] [--count 960] [--rounds 3] [--device N] [--scores out.json]
//
// REF/DIS: raw planar YUV (ffmpeg -f rawvideo -pix_fmt yuv420p10le, ...), little-endian 16-bit samples above 8 bit. The first --frames pairs are read
// into memory; a first pass scores each once (untimed, its scores written to --scores as JSON for comparison), then each round scores --count pairs,
// cycling through them, from add() of the first to finish(). Prints frames per second per round and the median, plus the scorer's own timing.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "vk_compute.h"
#include "vk_vmaf.h"

static void usage() {
	std::fprintf(stderr, "usage: ves-vmaf-bench --ref REF.yuv --dis DIS.yuv --size WxH [--bits 10] [--chroma 420|422|444] --model M.json [--model ...]\n"
	                     "                      [--frames 24] [--count 960] [--rounds 3] [--device N|auto|name] [--scores out.json]\n");
}

int main(int argc, char** argv) {
	std::string ref_path, dis_path, device = "auto", scores_path;
	std::vector<std::string> models;
	int w = 0, h = 0, bits = 10, chroma = 420, frames = 24, count = 960, rounds = 3;
	for (int i = 1; i < argc; i++) {
		const std::string a = argv[i];
		auto next = [&]() -> std::string {
			if (i + 1 >= argc) { usage(); std::exit(2); }
			return argv[++i];
		};
		if (a == "--ref") ref_path = next();
		else if (a == "--dis") dis_path = next();
		else if (a == "--size") { const std::string s = next(); if (std::sscanf(s.c_str(), "%dx%d", &w, &h) != 2) { usage(); return 2; } }
		else if (a == "--bits") bits = std::atoi(next().c_str());
		else if (a == "--chroma") chroma = std::atoi(next().c_str());
		else if (a == "--model") models.push_back(next());
		else if (a == "--frames") frames = std::atoi(next().c_str());
		else if (a == "--count") count = std::atoi(next().c_str());
		else if (a == "--rounds") rounds = std::atoi(next().c_str());
		else if (a == "--device") device = next();
		else if (a == "--scores") scores_path = next();
		else { usage(); return 2; }
	}
	if (ref_path.empty() || dis_path.empty() || w <= 0 || h <= 0 || models.empty() || frames <= 0 || count <= 0 || rounds <= 0 ||
	    (bits != 8 && bits != 10) || (chroma != 420 && chroma != 422 && chroma != 444)) {
		usage();
		return 2;
	}

	const int bps = bits > 8 ? 2 : 1;
	const int cw = chroma == 444 ? w : (w + 1) / 2, ch = chroma == 420 ? (h + 1) / 2 : h;
	const size_t ysize = (size_t)w * h * bps, csize = (size_t)cw * ch * bps, fsize = ysize + 2 * csize;
	auto load = [&](const std::string& path, std::vector<uint8_t>& out) -> int {
		std::ifstream f(path, std::ios::binary);
		if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
		out.resize(fsize * frames);
		f.read((char*)out.data(), (std::streamsize)out.size());
		return (int)(f.gcount() / (std::streamsize)fsize);
	};
	std::vector<uint8_t> ref, dis;
	const int nr = load(ref_path, ref), nd = load(dis_path, dis);
	const int n = std::min(nr, nd);
	if (n <= 0) { std::fprintf(stderr, "no whole frames of %dx%d (%d bit, %d) in the files\n", w, h, bits, chroma); return 1; }
	if (n < frames) std::fprintf(stderr, "only %d frame pairs in the files\n", n);

	std::string why;
	const int dev = GpuVmaf::pick_device(device, 0, -1, why);
	if (dev < 0) { std::fprintf(stderr, "no GPU: %s\n", why.c_str()); return 1; }
	GpuVmafFrame f;
	f.w = f.enc_w = w;
	f.h = f.enc_h = h;
	f.depth = f.enc_depth = bits;
	f.chroma = chroma;
	if (!GpuVmaf::supported(models, dev, f, why)) { std::fprintf(stderr, "not supported: %s\n", why.c_str()); return 1; }
	std::printf("device %d (%s), %dx%d %d bit %d, %d frame pairs in memory, %zu model(s)\n", dev, vulkan_device_name(dev).c_str(), w, h, bits, chroma, n,
	            models.size());

	auto run = [&](int pairs, std::vector<std::vector<double>>* scores, std::vector<std::vector<std::map<std::string, double>>>* features,
	               std::unique_ptr<GpuVmaf>* keep) -> double {
		auto s = GpuVmaf::create(models, dev, f, why);
		if (!s) { std::fprintf(stderr, "scorer: %s\n", why.c_str()); std::exit(1); }
		const auto t0 = std::chrono::steady_clock::now();
		for (int i = 0; i < pairs; i++) {
			const int k = i % n;
			const uint8_t* r[3] = {&ref[fsize * k], &ref[fsize * k + ysize], &ref[fsize * k + ysize + csize]};
			const uint8_t* d[3] = {&dis[fsize * k], &dis[fsize * k + ysize], &dis[fsize * k + ysize + csize]};
			const size_t ys = (size_t)w * bps, cs = (size_t)cw * bps;
			const size_t st[3] = {ys, cs, cs};
			if (!s->add(r, st, d, st)) { std::fprintf(stderr, "add failed at pair %d\n", i); std::exit(1); }
		}
		std::vector<std::vector<double>> sc;
		if (!s->finish(scores ? *scores : sc, features)) { std::fprintf(stderr, "finish failed\n"); std::exit(1); }
		const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		if (keep) *keep = std::move(s);
		return secs;
	};

	// First pass: every pair once, scores and features kept for comparison.
	std::vector<std::vector<double>> scores;
	std::vector<std::vector<std::map<std::string, double>>> features;
	run(n, &scores, &features, nullptr);
	if (!scores_path.empty()) {
		std::FILE* o = std::fopen(scores_path.c_str(), "w");
		if (!o) { std::fprintf(stderr, "cannot write %s\n", scores_path.c_str()); return 1; }
		std::fprintf(o, "{\"models\": [");
		for (size_t m = 0; m < models.size(); m++) {
			std::fprintf(o, "%s{\"model\": \"", m ? ", " : "");
			for (char c : models[m]) std::fprintf(o, c == '\\' || c == '"' ? "\\%c" : "%c", c);
			std::fprintf(o, "\", \"frames\": [");
			for (size_t i = 0; i < scores[m].size(); i++) {
				std::fprintf(o, "%s{\"vmaf\": %.17g", i ? ", " : "", scores[m][i]);
				if (m < features.size() && i < features[m].size())
					for (auto& [k, v] : features[m][i]) std::fprintf(o, ", \"%s\": %.17g", k.c_str(), v);
				std::fprintf(o, "}");
			}
			std::fprintf(o, "]}");
		}
		std::fprintf(o, "]}\n");
		std::fclose(o);
	}
	double mean = 0;
	for (double v : scores[0]) mean += v;
	std::printf("first model, mean VMAF of the %d pairs: %.6f\n", n, mean / (double)scores[0].size());

	std::vector<double> fps;
	std::unique_ptr<GpuVmaf> last;
	for (int r = 0; r < rounds; r++) {
		const double secs = run(count, nullptr, nullptr, &last);
		fps.push_back(count / secs);
		std::printf("round %d: %d pairs in %.3f s = %.1f fps\n", r + 1, count, secs, count / secs);
	}
	std::sort(fps.begin(), fps.end());
	std::printf("median: %.1f fps\n", fps[fps.size() / 2]);
	double copy, wait, gpu, results;
	last->timing(copy, wait, gpu, results);
	std::printf("last round, ms per pair: copy in %.3f, waiting %.3f, GPU %.3f, results %.3f\n", 1e3 * copy / count, 1e3 * wait / count, 1e3 * gpu / count,
	            1e3 * results / count);
	std::vector<std::pair<std::string, double>> parts;
	last->gpu_parts(parts);
	for (auto& [name, secs] : parts) std::printf("  GPU %-24s %.3f ms per pair\n", name.c_str(), 1e3 * secs / count);
	uint64_t dl, host, per;
	int inflight;
	last->memory(dl, host, per, inflight);
	std::printf("memory: %.1f MB VRAM, %.1f MB system, %.1f MB per frame in flight (%d)\n", dl / 1e6, host / 1e6, per / 1e6, inflight);
	return 0;
}
