#include "vmaf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <regex>
#include <set>
#include <sstream>

#include "decode.h"
#include "json.h"
#include "vk_compute.h"
#include "vk_vmaf.h"

extern "C" {
#include <libvmaf/feature.h>
#include <libvmaf/libvmaf.h>
#include <libvmaf/model.h>
#include <libvmaf/picture.h>
}

namespace {

// Loads `model` ("path=<file>" or "version=<built-in>") into `out`.
// `name` is what libvmaf files the scores under: models scored in one context need names of their own, or every one gets the first one's scores.
int load_model(const std::string& model, VmafModel** out, const char* name = "vmaf") {
	VmafModelConfig mc{name, 0};
	const bool from_path = model.rfind("path=", 0) == 0;
	const std::string spec = model.substr(model.find('=') == std::string::npos ? 0 : model.find('=') + 1);
	return from_path ? vmaf_model_load_from_path(out, &mc, spec.c_str()) : vmaf_model_load(out, &mc, spec.c_str());
}

// The feature names of a model file; empty for a built-in model or a file that does not read.
std::vector<std::string> model_features(const std::string& model) {
	static std::mutex m;
	static std::map<std::string, std::vector<std::string>> known;
	std::lock_guard lock(m);
	if (auto it = known.find(model); it != known.end()) return it->second;
	std::vector<std::string> names;
	if (model.rfind("path=", 0) == 0) {
		try {
			std::ifstream f(model.substr(5), std::ios::binary);
			std::stringstream ss; ss << f.rdbuf();
			const Json j = Json::parse(ss.str());
			if (const Json* md = j.find("model_dict")) if (const Json* fn = md->find("feature_names")) for (auto& x : fn->a) names.push_back(x.text());
		} catch (const std::exception&) {}
	}
	return known[model] = names;
}

// libvmaf's names of a model's SpEED chroma U and V features ("speed_chroma_u" and the options that are not the default, as
// feature_name.c builds them: sorted by option name, by alias, doubles as %g); empty if the model has no SpEED.
std::pair<std::string, std::string> speed_feature_names(const std::string& model) {
	if (model.rfind("path=", 0) != 0) return {};
	try {
		std::ifstream f(model.substr(5), std::ios::binary);
		std::stringstream ss; ss << f.rdbuf();
		const Json j = Json::parse(ss.str());
		const Json* md = j.find("model_dict");
		const Json* fn = md ? md->find("feature_names") : nullptr;
		const Json* fo = md ? md->find("feature_opts_dicts") : nullptr;
		if (!fn) return {};
		for (size_t i = 0; i < fn->a.size(); i++) {
			if (fn->a[i].text().rfind("Speed_chroma", 0) != 0) continue;
			struct Opt { const char* name; const char* alias; double def; const char* sdef; bool integer; };
			static const Opt opts[] = { // alphabetical by name
				{"speed_kernelscale", "ks", 1.0, nullptr, false}, {"speed_max_val", "mxv", 1000.0, nullptr, false}, {"speed_nn_floor", "nnf", 0.0, nullptr, false},
				{"speed_prescale", "ps", 1.0, nullptr, false}, {"speed_prescale_method", "psm", 0, "nearest", false}, {"speed_sigma_nn", "snn", 0.29, nullptr, false},
				{"speed_weight_var_mode", "wvm", 0, nullptr, true}};
			std::string suffix;
			const Json* o = fo && i < fo->a.size() ? &fo->a[i] : nullptr;
			for (const Opt& op : opts) {
				const Json* v = o ? o->find(op.name) : nullptr;
				if (!v) continue;
				char b[64];
				if (op.sdef) { if (v->text() == op.sdef) continue; snprintf(b, sizeof b, "%s", v->text().c_str()); }
				else if (op.integer) { const int x = atoi(v->text().c_str()); if (x == (int)op.def) continue; snprintf(b, sizeof b, "%d", x); }
				else { const double x = atof(v->text().c_str()); if (x == op.def) continue; snprintf(b, sizeof b, "%g", x); }
				suffix += std::string("_") + op.alias + "_" + b;
			}
			return {"speed_chroma_u" + suffix, "speed_chroma_v" + suffix};
		}
	} catch (const std::exception&) {}
	return {};
}

} // namespace

std::vector<std::string> vmaf_models(const VmafOptions& o) {
	std::vector<std::string> all{o.model};
	for (auto& m : o.extra_models) if (std::find(all.begin(), all.end(), m) == all.end()) all.push_back(m);
	return all;
}

bool vmaf_check_model(const std::string& model, std::string& why) {
	VmafContext* ctx = nullptr;
	VmafModel* m = nullptr;
	VmafConfiguration cfg{};
	cfg.log_level = VMAF_LOG_LEVEL_NONE;
	cfg.n_threads = 1;
	bool ok = false;
	const std::string version = vmaf_version();
	if (vmaf_init(&ctx, cfg) < 0) why = "libvmaf does not start";
	else if (load_model(model, &m) < 0) why = "libvmaf " + version + " cannot load the VMAF model '" + model + "'";
	else if (vmaf_use_features_from_model(ctx, m) < 0) {
		why = "libvmaf " + version + " lacks a feature the VMAF model '" + model + "' uses";
		for (auto& f : model_features(model)) if (f.rfind("Speed_chroma", 0) == 0) { why += " (VMAF v1 needs libvmaf 3.2.1 or later, or 3.2.0 built with -Denable_float=true)"; break; }
	} else ok = true;
	if (ctx) vmaf_close(ctx);
	if (m) vmaf_model_destroy(m);
	return ok;
}

int vmaf_model_min_depth(const std::string& model) {
	for (auto& f : model_features(model)) if (f.rfind("Cambi_", 0) == 0) return 10;
	return 8;
}

std::string vmaf_model_display_name(const std::string& model) {
	if (model.rfind("path=", 0) != 0) return model.substr(model.find('=') == std::string::npos ? 0 : model.find('=') + 1);
	return std::filesystem::path(model.substr(5)).stem().string();
}

std::string vmaf_model_generation(const std::string& model) {
	// By the features: v1 brought CAMBI and the chroma feature; v0 models use adm2/motion2/vif. Else by the name (vmaf_v0.6.1, vmaf_4k_v0.6.1neg, vmaf_v1.0.16_3d0h).
	for (auto& f : model_features(model)) if (f.rfind("Cambi_", 0) == 0 || f.rfind("Speed_chroma", 0) == 0) return "v1";
	for (auto& f : model_features(model)) if (f.find("adm2") != std::string::npos) return "v0";
	std::smatch mt;
	const std::string name = vmaf_model_display_name(model);
	if (std::regex_search(name, mt, std::regex("(^|[^0-9a-z])v([0-9]+)\\.[0-9]", std::regex::icase))) return "v" + mt[2].str();
	return "";
}

namespace {

// What CAMBI (VMAF v1's banding feature) is told about the encode: it looks for banding at the encoded size and bit depth (as Netflix's model docs
// ask for), instead of in the upscaled frames. Its limits: 144..7680 samples and at least 216 one way, 6..16 bits, never more than what is scored,
// and not smaller one way but larger the other; outside them it uses what is scored. Both scorers are given these values.
struct CambiEncode { int w, h, depth; };
CambiEncode cambi_encode(const Shape& enc, int w, int h, int depth) {
	CambiEncode c{w, h, depth};
	if ((enc.w < w || enc.h < h) && enc.w <= w && enc.h <= h && enc.w >= 144 && enc.h >= 144 && (enc.w >= 216 || enc.h >= 216)) { c.w = enc.w; c.h = enc.h; }
	if (enc.pix.depth < depth && enc.pix.depth >= 6) c.depth = enc.pix.depth;
	return c;
}

// Scores picture pairs; every score stays in memory.
struct Scorer {
	virtual ~Scorer() = default;
	virtual bool add(const Picture& ref, const Picture& dis) = 0;
	// Per model: the scores and, for a model with SpEED chroma, its U and V values (else empty).
	virtual bool finish(std::vector<std::vector<double>>& scores, std::vector<std::vector<double>>& u, std::vector<std::vector<double>>& v) = 0;
	// The GPU scorer's parts of add() and finish() (copying in, waiting for a frame in flight) and its background work (GPU, results); see GpuVmaf::timing().
	virtual bool timing(double& copy, double& wait, double& gpu, double& results) const { copy = wait = gpu = results = 0; return false; }
	virtual std::string memory() const { return {}; }
	virtual void speed_timing(double parts[3]) const { parts[0] = parts[1] = parts[2] = 0; }
};

// Where a scoring pass spends its time, in seconds, for VES_SCORE_TIMING=1: one line per pass when it ends. Per thread: a pass runs on one.
struct ScoreClock {
	double decode = 0, packets = 0, reference = 0, convert = 0, score = 0;
	std::chrono::steady_clock::time_point began;
};
thread_local ScoreClock score_clock;
struct Timed { // adds the time from here to the end of the scope to `to`
	double& to;
	std::chrono::steady_clock::time_point from = std::chrono::steady_clock::now();
	~Timed() { to += std::chrono::duration<double>(std::chrono::steady_clock::now() - from).count(); }
};

// With libvmaf, in-process, on the CPU.
class CpuScorer : public Scorer {
	VmafContext* ctx = nullptr;
	std::vector<VmafModel*> models;
	std::vector<std::pair<std::string, std::string>> speed_names;
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
	~CpuScorer() override {
		if (ctx) vmaf_close(ctx);
		for (auto* m : models) vmaf_model_destroy(m);
	}

	// `w` x `h` at `depth` bits is what is scored; `enc` is what was encoded (see cambi_encode()).
	bool open(const VmafOptions& o, const Shape& enc, int w, int h, int depth) {
		VmafConfiguration cfg{};
		cfg.log_level = VMAF_LOG_LEVEL_WARNING;
		cfg.n_threads = o.threads;
		if (vmaf_init(&ctx, cfg) < 0) { fprintf(stderr, "vmaf: init failed\n"); return false; }
		const CambiEncode ce = cambi_encode(enc, w, h, depth);
		// Every model's features go into the one context; libvmaf computes a feature with the same options once.
		for (const std::string& spec : vmaf_models(o)) {
			VmafModel* model = nullptr;
			const std::string name = "vmaf_" + std::to_string(models.size());
			if (load_model(spec, &model, name.c_str()) < 0) { fprintf(stderr, "vmaf: cannot load model '%s'\n", spec.c_str()); return false; }
			models.push_back(model);
			speed_names.push_back(speed_feature_names(spec));
			VmafFeatureDictionary* cambi = nullptr;
			if (ce.w != w || ce.h != h) {
				vmaf_feature_dictionary_set(&cambi, "enc_width", std::to_string(ce.w).c_str());
				vmaf_feature_dictionary_set(&cambi, "enc_height", std::to_string(ce.h).c_str());
			}
			if (ce.depth != depth) vmaf_feature_dictionary_set(&cambi, "enc_bitdepth", std::to_string(ce.depth).c_str());
			if (cambi && vmaf_model_feature_overload(model, "cambi", cambi) < 0) { fprintf(stderr, "vmaf: cannot pass the encoded size to CAMBI\n"); return false; }
			if (vmaf_use_features_from_model(ctx, model) < 0) { fprintf(stderr, "vmaf: features of model '%s' unavailable\n", spec.c_str()); return false; }
		}
		return true;
	}

	bool add(const Picture& ref, const Picture& dis) override {
		VmafPicture rp{}, dp{};
		if (!to_picture(ref, rp)) return false;
		if (!to_picture(dis, dp)) { vmaf_picture_unref(&rp); return false; }
		if (vmaf_read_pictures(ctx, &rp, &dp, count) < 0) { fprintf(stderr, "vmaf: scoring frame %u failed\n", count); return false; }
		count++;
		return true;
	}

	bool finish(std::vector<std::vector<double>>& scores, std::vector<std::vector<double>>& u, std::vector<std::vector<double>>& v) override {
		if (vmaf_read_pictures(ctx, nullptr, nullptr, 0) < 0) { fprintf(stderr, "vmaf: flushing failed\n"); return false; }
		const size_t n = models.size();
		scores.assign(n, std::vector<double>(count, 0));
		u.assign(n, {}); v.assign(n, {});
		for (size_t k = 0; k < n; k++) {
			for (unsigned i = 0; i < count; i++) {
				if (vmaf_score_at_index(ctx, models[k], &scores[k][i], i) < 0) { fprintf(stderr, "vmaf: no score for frame %u\n", i); return false; }
			}
			if (speed_names[k].first.empty()) continue;
			std::vector<double> uu(count), vv(count);
			bool got = true;
			for (unsigned i = 0; got && i < count; i++)
				got = vmaf_feature_score_at_index(ctx, speed_names[k].first.c_str(), &uu[i], i) >= 0 && vmaf_feature_score_at_index(ctx, speed_names[k].second.c_str(), &vv[i], i) >= 0;
			if (got) { u[k] = std::move(uu); v[k] = std::move(vv); }
			else {
				static std::atomic<bool> told{false};
				if (!told.exchange(true)) fprintf(stderr, "vmaf: the chroma values (%s) are not where they were expected; results go without them\n", speed_names[k].first.c_str());
			}
		}
		return true;
	}
};

// On the GPU (Vulkan compute, vk_vmaf.cpp): 8 and 10 bit planar YUV; the chroma planes are only read by models with a chroma feature. All
// models in one pass.
class GpuScorer : public Scorer {
	std::unique_ptr<GpuVmaf> gpu;
	int w, h;

public:
	GpuScorer(std::unique_ptr<GpuVmaf> g, int w, int h) : gpu(std::move(g)), w(w), h(h) {}
	bool add(const Picture& ref, const Picture& dis) override {
		if (ref.width != w || ref.height != h || dis.width != w || dis.height != h) { fprintf(stderr, "vmaf: the frames are not %dx%d\n", w, h); return false; }
		if (ref.fmt != dis.fmt || !ref.fmt.planar || ref.fmt.rgb) { fprintf(stderr, "vmaf: the frames are not in the same planar YUV format\n"); return false; }
		const uint8_t* rd[3] = {ref.data[0], ref.data[1], ref.data[2]};
		const uint8_t* dd[3] = {dis.data[0], dis.data[1], dis.data[2]};
		const size_t rs[3] = {(size_t)ref.stride[0], (size_t)ref.stride[1], (size_t)ref.stride[2]};
		const size_t ds[3] = {(size_t)dis.stride[0], (size_t)dis.stride[1], (size_t)dis.stride[2]};
		return gpu->add(rd, rs, dd, ds);
	}
	bool timing(double& copy, double& wait, double& g, double& results) const override { gpu->timing(copy, wait, g, results); return true; }
	void speed_timing(double parts[3]) const override { gpu->speed_timing(parts); }
	std::string memory() const override {
		uint64_t device, host, per_frame;
		int frames;
		gpu->memory(device, host, per_frame, frames);
		char b[160];
		snprintf(b, sizeof b, "; GPU scorer memory %.0f MB device local, %.0f MB other, of which %d frames in flight %.1f MB each", device / 1048576.0, host / 1048576.0, frames, per_frame / 1048576.0);
		return b;
	}
	bool finish(std::vector<std::vector<double>>& scores, std::vector<std::vector<double>>& u, std::vector<std::vector<double>>& v) override {
		std::vector<std::vector<std::map<std::string, double>>> fe;
		if (!gpu->finish(scores, &fe)) return false;
		u.assign(scores.size(), {}); v.assign(scores.size(), {});
		for (size_t k = 0; k < fe.size(); k++) {
			if (fe[k].empty() || !fe[k][0].count("Speed_chroma_feature_speed_chroma_u_score")) continue;
			for (auto& f : fe[k]) { u[k].push_back(f.at("Speed_chroma_feature_speed_chroma_u_score")); v[k].push_back(f.at("Speed_chroma_feature_speed_chroma_v_score")); }
		}
		return true;
	}
};

} // namespace

bool vmaf_score(const RefInfo& ref, const FrameFeed& next_ref, Codec codec, const PacketFeed& next_packet, const std::vector<uint8_t>& extradata, const Shape& enc,
                const Color& decoded_color, const ShownFeed& shown, Fmt score_fmt, const VmafOptions& opt, const ScaleOptions& scale, std::vector<double>& scores, RunProgress* rp,
                VmafDetail* detail) {
	score_clock = ScoreClock();
	score_clock.began = std::chrono::steady_clock::now();
	auto decoder = make_decoder();
	if (!decoder->open(codec, extradata)) return false;
	long long m = 0;
	bool flushed = false;
	const FrameFeed next_dis = [&](Picture& d) -> int {
		for (;;) {
			int got;
			{ Timed t{score_clock.decode}; got = decoder->receive(d); }
			if (got < 0) { fprintf(stderr, "vmaf: decoding the encoded stream failed after %lld frames\n", m); return -1; }
			if (got == 1) { m++; return 1; }
			if (flushed) return 0;
			std::vector<uint8_t> packet;
			bool more;
			{ Timed t{score_clock.packets}; more = next_packet(packet); }
			if (!more) { decoder->flush(); flushed = true; continue; }
			bool sent;
			{ Timed t{score_clock.decode}; sent = decoder->send(packet.data(), packet.size()); }
			if (!sent) { fprintf(stderr, "vmaf: the encoded stream does not decode (packet of %zu bytes, after %lld frames)\n", packet.size(), m); return -1; }
		}
	};
	return vmaf_score_frames(ref, next_ref, next_dis, enc, decoded_color, shown, score_fmt, opt, scale, scores, rp, detail);
}

bool vmaf_score_frames(const RefInfo& ref, const FrameFeed& next_ref, const FrameFeed& next_dis, const Shape& enc, const Color& decoded_color, const ShownFeed& shown,
                       Fmt score_fmt, const VmafOptions& opt, const ScaleOptions& scale, std::vector<double>& scores, RunProgress* rp, VmafDetail* detail) {
	if (rp) rp->begin(2, std::max<long long>(1, ref.frames));

	const bool convert_dis = enc.w != ref.width || enc.h != ref.height || enc.pix != score_fmt || (decoded_color.range ? decoded_color.range : 1) != ref.score_range; // scale back, change format, change range back

	// "gpu" needs the GPU scorer; "auto" falls back to libvmaf when it cannot be used.
	std::unique_ptr<Scorer> scorer;
	const std::vector<std::string> models = vmaf_models(opt);
	bool ok = true;
	if (opt.device != "cpu") {
		std::string why;
		std::unique_ptr<GpuVmaf> gpu;
		if ((score_fmt.depth != 8 && score_fmt.depth != 10) || !score_fmt.planar || score_fmt.rgb) why = "only 8 and 10 bit YUV is supported";
		else if (std::any_of(models.begin(), models.end(), [](const std::string& m) { return m.rfind("path=", 0) != 0; })) why = "a model is not a file";
		else if (const int device = GpuVmaf::pick_device(opt.gpu, opt.encoder_vendor, opt.encoder_gpu, why); device >= 0) {
			GpuVmafFrame f;
			f.w = ref.width; f.h = ref.height; f.depth = score_fmt.depth; f.chroma = score_fmt.chroma;
			const CambiEncode ce = cambi_encode(enc, ref.width, ref.height, score_fmt.depth);
			f.enc_w = ce.w; f.enc_h = ce.h; f.enc_depth = ce.depth;
			std::vector<std::string> paths;
			for (auto& m : models) paths.push_back(m.substr(5));
			gpu = GpuVmaf::create(paths, device, f, why);
			if (gpu) { // say once for each device which one scores
				static std::mutex m;
				static std::set<int> told;
				std::lock_guard lock(m);
				std::string e;
				if (told.insert(device).second) fprintf(stderr, "vmaf: scoring on Vulkan device %d (%s)\n", device, Vk::devices(e)[device].name.c_str());
			}
		}
		if (gpu) scorer = std::make_unique<GpuScorer>(std::move(gpu), ref.width, ref.height);
		else if (opt.device == "gpu") { fprintf(stderr, "vmaf: GPU scoring is not possible: %s\n", why.c_str()); ok = false; }
		else {
			static std::atomic<bool> told{false};
			if (!told.exchange(true)) fprintf(stderr, "vmaf: GPU scoring is not used (%s); libvmaf scores on the CPU\n", why.c_str());
		}
	}
	if (ok && !scorer) {
		auto cpu = std::make_unique<CpuScorer>();
		ok = cpu->open(opt, enc, ref.width, ref.height, score_fmt.depth);
		scorer = std::move(cpu);
	}
	auto up = make_converter();
	Picture d, u, r;
	long long m = 0, mo = 0, k = 0; // decoded frames in, converted frames out, original frame index

	// Decoded frame mi (at the original resolution) is on screen for every original frame up to the next one. It is scored once, against the original
	// frame it was made from (the first of them); the original frames the encode dropped (a lower frame rate) show that same frame again, so they get its
	// score instead of one against a picture that moved on. `repeats` counts the original frames of each scored frame.
	std::vector<uint32_t> repeats;
	auto show = [&](const Picture& pic, int64_t mi) {
		uint32_t on_screen;
		bool first = true;
		while (ok && shown((size_t)k, on_screen) && on_screen == mi) {
			int rr;
			{ Timed t{score_clock.reference}; rr = next_ref(r); }
			if (rr < 0) { fprintf(stderr, "vmaf: the original frames stopped coming (frame %lld)\n", k); ok = false; break; }
			if (rr == 0) break;
			if (!first) { repeats.back()++; k++; if (rp) rp->step(); continue; }
			first = false;
			repeats.push_back(1);
			{ Timed t{score_clock.score}; ok = scorer->add(r, pic); }
			if (!ok) fprintf(stderr, "vmaf: the scorer refused frame %lld (original %dx%d %d-bit %d%s, encoded %dx%d %d-bit %d%s)\n", k, r.width, r.height, r.fmt.depth, r.fmt.chroma, r.fmt.planar ? "" : " packed", pic.width, pic.height, pic.fmt.depth, pic.fmt.chroma, pic.fmt.planar ? "" : " packed");
			k++;
			if (rp) rp->step();
		}
	};
	// libplacebo holds frames back until it has seen the next one, so converted frames are collected as they appear.
	auto collect_converted = [&] {
		int got;
		while (ok && (got = up->pop(u)) == 1) show(u, mo++);
		if (ok && got < 0) fprintf(stderr, "vmaf: scaling the encoded frames back failed\n");
		ok = ok && got >= 0;
	};
	for (int got; ok && (got = next_dis(d)) != 0;) {
		if (got < 0) { ok = false; break; }
		// AMF's AV1 on RDNA 3 codes 1080 lines as 1082 (Av1AlignmentMode 64X16_1080P_CODED_1082); a few padding lines or columns at the bottom and
		// right are cut off, as the frame's render size says.
		if (d.fmt == enc.pix && d.width >= enc.w && d.height >= enc.h && d.width - enc.w < 64 && d.height - enc.h < 16) { d.width = enc.w; d.height = enc.h; }
		if (d.width != enc.w || d.height != enc.h || d.fmt != enc.pix) {
			fprintf(stderr, "decoded stream is %dx%d %d-bit %d, expected %dx%d %d-bit %d\n", d.width, d.height, d.fmt.depth, d.fmt.chroma, enc.w, enc.h, enc.pix.depth, enc.pix.chroma);
			ok = false; break;
		}
		// Encoders do not necessarily write the colour description, so use the one the encoder was given before converting.
		d.color = decoded_color;
		if (!convert_dis) show(d, m);
		else {
			d.pts = m;
			d.duration = 1; // libplacebo needs it to know when the last frame ends; the packets carry none
			if (m == 0 && !up->open(d, enc.fps.inverse(), ref.width, ref.height, score_fmt, scale, ref.score_range)) { fprintf(stderr, "vmaf: cannot scale the encoded frames back to %dx%d\n", ref.width, ref.height); ok = false; break; }
			bool pushed;
			{ Timed t{score_clock.convert}; pushed = up->push(&d); }
			if (!pushed) { fprintf(stderr, "scaling the decoded frame back failed\n"); ok = false; break; }
			collect_converted();
		}
		m++;
	}
	if (ok && convert_dis) { ok = up->push(nullptr); collect_converted(); }
	if (ok && k == 0) { fprintf(stderr, "no frames were decoded\n"); ok = false; }
	uint32_t extra;
	if (ok && (shown((size_t)k, extra) || next_ref(r) == 1)) { fprintf(stderr, "encoded stream does not cover the reference frames\n"); ok = false; }
	if (!ok) return false;
	std::vector<std::vector<double>> all, cu, cv;
	bool finished;
	{ Timed t{score_clock.score}; finished = scorer->finish(all, cu, cv); } // the GPU scorer's last frames in flight
	if (!finished || all.empty()) { fprintf(stderr, "vmaf: the scorer could not finish\n"); return false; }
	if (const char* e = getenv("VES_SCORE_TIMING"); e && atoi(e) > 0 && k > 0) {
		const ScoreClock& c = score_clock;
		double copy, wait, gpu, results;
		const bool split = scorer->timing(copy, wait, gpu, results);
		const double all_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - c.began).count(), per = 1000.0 / (double)m;
		char b[512];
		// score = copy in + waiting (+ a little); GPU and results run alongside, several frames at once, and only show in score through the waiting
		double sp[3];
		scorer->speed_timing(sp);
		char sb[160] = "";
		if (sp[0] + sp[1] + sp[2] > 0) snprintf(sb, sizeof sb, " (of which SpEED %.2f: blur and covariances %.2f, eigenvalues and QR %.2f, blocks and score %.2f)", (sp[0] + sp[1] + sp[2]) * per, sp[0] * per, sp[1] * per, sp[2] * per);
		if (split) snprintf(b, sizeof b, " (copy in %.2f, waiting for frames in flight %.2f; alongside: GPU %.2f%s, results %.2f)", copy * per, wait * per, gpu * per, sb, results * per);
		else b[0] = 0;
		std::string sc = b;
		fprintf(stderr, "vmaf timing, %s %dx%d %d-bit %d, %lld frames, %.1f fps: ms per frame: decode %.2f, waiting for packets %.2f, reference %.2f, convert %.2f, score %.2f%s, rest %.2f%s\n",
		        rp ? "true score" : "distorted score", ref.width, ref.height, score_fmt.depth, score_fmt.chroma, m, m / all_s, c.decode * per, c.packets * per, c.reference * per, c.convert * per, c.score * per, sc.c_str(),
		        (all_s - c.decode - c.packets - c.reference - c.convert - c.score) * per, scorer->memory().c_str());
	}
	// One value per original frame: a frame the encode dropped repeats the score of the frame on screen.
	auto expand = [&](std::vector<double>& v) {
		if (v.empty() || v.size() != repeats.size() || (long long)repeats.size() == k) return;
		std::vector<double> out;
		out.reserve((size_t)k);
		for (size_t i = 0; i < v.size(); i++) out.insert(out.end(), repeats[i], v[i]);
		v = std::move(out);
	};
	for (auto* list : {&all, &cu, &cv}) for (auto& v : *list) expand(v);
	scores = std::move(all[0]);
	if (detail) {
		detail->chroma_u = std::move(cu[0]); detail->chroma_v = std::move(cv[0]);
		detail->extra.clear();
		for (size_t k = 1; k < all.size(); k++) detail->extra.push_back(ModelScores{std::move(all[k]), std::move(cu[k]), std::move(cv[k])});
	}
	return true;
}
