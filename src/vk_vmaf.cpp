#define _USE_MATH_DEFINES
#include "vk_vmaf.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>

#include "gpu.h"
#include "vk_compute.h"

static std::string lower(std::string s) {
	for (auto& c : s) c = (char)std::tolower((unsigned char)c);
	return s;
}


// The Vulkan device that is the n-th GPU of `vendor` as gpu.h lists them (what gpu=n means to that vendor's encoder): the one with the same LUID where the
// driver says, else the n-th device of that vendor in Vulkan's order if it is the same model, else the first device of that model. -1 if none matches.
static int device_of_encoder_gpu(const std::vector<VkDeviceInfo>& all, unsigned vendor, int n) {
	const auto gpus = list_gpus(vendor);
	if (n < 0 || n >= (int)gpus.size()) return -1;
	const GpuInfo& g = gpus[n];
	if (g.luid) for (auto& d : all) if (d.luid_valid && d.luid == g.luid) return d.index;
	int seen = 0, same_model = -1;
	for (auto& d : all) {
		if (d.vendor_id != vendor) continue;
		if (d.device_id == g.device_id) {
			if (seen == n) return d.index;
			if (same_model < 0) same_model = d.index;
		}
		seen++;
	}
	return same_model;
}

int vulkan_device_of_gpu(unsigned vendor, int n) {
	std::string why;
	return device_of_encoder_gpu(Vk::devices(why), vendor, n);
}

std::string vulkan_device_name(int index) {
	std::string why;
	for (auto& d : Vk::devices(why)) if (d.index == index) return d.name;
	return {};
}

std::vector<VkDeviceInfo> vulkan_devices_fastest_first() {
	std::string why;
	auto all = Vk::devices(why);
	std::stable_sort(all.begin(), all.end(), [](const VkDeviceInfo& a, const VkDeviceInfo& b) {
		if (a.discrete != b.discrete) return a.discrete;
		return a.local_memory > b.local_memory;
	});
	return all;
}

bool gpu_of_vulkan_device(int index, unsigned& vendor, int& n) {
	for (unsigned v : {VENDOR_NVIDIA, VENDOR_AMD, VENDOR_INTEL}) {
		const size_t count = list_gpus(v).size();
		for (size_t i = 0; i < count; i++)
			if (vulkan_device_of_gpu(v, (int)i) == index) { vendor = v; n = (int)i; return true; }
	}
	return false;
}

// The fastest device the scorer can use, -1 if there is none.
static int fastest_usable() {
	for (auto& d : vulkan_devices_fastest_first()) if (d.usable) return d.index;
	return -1;
}

int GpuVmaf::pick_device(const std::string& spec, unsigned encoder_vendor, int encoder_gpu, std::string& why) {
	const auto all = Vk::devices(why);
	if (all.empty()) { if (why.empty()) why = "there is no Vulkan device"; return -1; }
	auto list = [&] {
		std::string s;
		for (auto& d : all) s += (s.empty() ? "" : ", ") + std::to_string(d.index) + ": " + d.name + (d.usable ? "" : " (unusable)");
		return s;
	};
	if (spec.empty() || spec == "auto") {
		// The fastest GPU, whichever GPU encodes: scoring does not need the encoder's GPU, and an integrated one is far too slow for it.
		const int d = fastest_usable();
		if (d < 0) why = "no Vulkan GPU with 64 bit integer and float support (Vulkan devices: " + list() + "; vmaf.gpu picks one)";
		return d;
	}
	if (spec == "own") { // the GPU that encodes, as before
		const int d = encoder_vendor ? device_of_encoder_gpu(all, encoder_vendor, encoder_gpu) : -1;
		if (d >= 0 && all[d].usable) return d;
		const int f = fastest_usable();
		if (f < 0) why = "no Vulkan GPU with 64 bit integer and float support (Vulkan devices: " + list() + ")";
		return f;
	}
	if (std::all_of(spec.begin(), spec.end(), [](char c) { return c >= '0' && c <= '9'; })) {
		const int d = atoi(spec.c_str());
		if (d < (int)all.size()) return d; // Vk::available() says why it cannot be used, if it cannot
		why = "there is no Vulkan device " + spec + " (Vulkan devices: " + list() + ")";
		return -1;
	}
	int fallback = -1;
	for (auto& d : all) {
		if (lower(d.name).find(lower(spec)) == std::string::npos) continue;
		if (d.usable) return d.index;
		if (fallback < 0) fallback = d.index;
	}
	if (fallback >= 0) return fallback;
	why = "no Vulkan device is named like '" + spec + "' (Vulkan devices: " + list() + ")";
	return -1;
}

#ifndef VES_NO_VULKAN
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <sstream>
#include <stdexcept>

#include "json.h"
#include "vk_compute.h"
#include "vmaf_speed.h"
#include "vmaf_tables.h"

// VMAF on Vulkan compute, ported from libvmaf 3.2.1's CPU code (integer_motion.c, integer_vif.c, integer_adm.c, cambi.c, speed.c, predict.c): the
// per-sample work in the shaders (shaders/), the sums and the model's score step on the CPU. VMAF v0 models use VIF, ADM (adm2) and motion2 on
// the luma plane; VMAF v1 models use adm3 (ADM with the additive impairment measure, AIM), motion3, CAMBI (banding, on the decoded luma plane at the
// encoded size) and SpEED on the chroma planes (whose last steps, on frames decimated by 16, run on the CPU: vmaf_speed.h).

namespace {

// The options of the features a model uses (libvmaf's names and defaults).
struct AdmOpts {
	double egl = 100.0, nvd = 3.0, csf_scale = 1.0, csf_diag_scale = 1.0, noise_weight = 0.03125, dlm_weight = 0.5, min_val = 0.0;
	int rdh = 1080, csf_mode = 0;
};
struct MotionOpts {
	double max_val = 10000.0, blend_factor = 1.0, blend_offset = 40.0, fps_weight = 1.0;
	bool five_frame = false, moving_average = false;
};
struct CambiOpts {
	double topk = 0.6, tvi = 0.019, vlt = 0.0, max_val = 1000.0;
	int window = 65, max_log_contrast = 2, high_res_speedup = 0;
	std::string eotf = "bt1886";
};
struct SpeedOpts {
	double kernelscale = 1.0, prescale = 1.0, sigma_nn = 0.29, nn_floor = 0.0, max_val = 1000.0;
	std::string prescale_method = "nearest";
	int weight_var_mode = 0;
};

// What a model needs, and its score step: normalise the features, nu-SVR with an RBF kernel (libsvm), denormalise, transform, clip.
struct Model {
	std::vector<std::string> names;
	std::vector<double> slopes, intercepts;
	std::vector<std::vector<double>> sv;
	std::vector<double> coef;
	double gamma = 0, rho = 0, clip_min = 0, clip_max = 100;
	bool clip = false;
	bool transform = false, has_p0 = false, has_p1 = false, has_p2 = false, out_lte_in = false, out_gte_in = false;
	double p0 = 0, p1 = 0, p2 = 0;
	double chroma_correction = 0; // VMAF v1: a chroma score of exactly 0 is replaced with one from adm3
	bool vif = false, adm = false, aim = false, motion = false, cambi = false, speed = false;
	double vif_egl = 100.0;
	AdmOpts adm_o;
	MotionOpts motion_o;
	CambiOpts cambi_o;
	SpeedOpts speed_o;

	double predict(std::vector<double> x) const { // x: raw values in `names` order
		const size_t n = names.size();
		for (size_t i = 0; i < n; i++) x[i] = slopes[i + 1] * x[i] + intercepts[i + 1];
		if (chroma_correction != 0) { // predict.c, post_process_feature_from_another()
			int guiding = -1, guided = -1;
			for (size_t i = 0; i < n; i++) {
				if (names[i].find("adm3") != std::string::npos) guiding = (int)i;
				if (names[i].find("speed_chroma") != std::string::npos) guided = (int)i;
			}
			if (guiding >= 0 && guided >= 0) {
				const double g = (x[guiding] - intercepts[guiding + 1]) / slopes[guiding + 1];
				const double d = (x[guided] - intercepts[guided + 1]) / slopes[guided + 1];
				if (d == 0.0) x[guided] = slopes[guided + 1] * ((-chroma_correction * g) + chroma_correction) + intercepts[guided + 1];
			}
		}
		double sum = 0;
		for (size_t k = 0; k < sv.size(); k++) {
			double d2 = 0;
			for (size_t i = 0; i < n; i++) { const double d = x[i] - sv[k][i]; d2 += d * d; }
			sum += coef[k] * exp(-gamma * d2);
		}
		sum -= rho;
		double y = (sum - intercepts[0]) / slopes[0];
		if (transform) { // polynomial mapping and rectification (predict.c, transform(); models with knots are refused)
			const double y_in = y;
			double out = y_in;
			if (has_p0 || has_p1 || has_p2) {
				out = 0.;
				if (has_p0) out += p0;
				if (has_p1) out += p1 * y_in;
				if (has_p2) out += p2 * y_in * y_in;
			}
			if (out_lte_in) out = (out > y_in) ? y_in : out;
			if (out_gte_in) out = (out < y_in) ? y_in : out;
			y = out;
		}
		if (clip) {
			y = y < clip_min ? clip_min : y;
			y = y > clip_max ? clip_max : y;
		}
		return y;
	}
};

std::string slurp(const std::string& path) {
	std::ifstream f(path, std::ios::binary);
	if (!f) throw std::runtime_error("cannot read " + path);
	return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

bool truthy(const Json& v) { if (v.type == Json::Null) return false; const std::string t = v.text(); return t == "true" || t == "1"; }
double num(const Json& v) { return atof(v.text().c_str()); }

Model load_model(const std::string& path) {
	Json j = Json::parse(slurp(path));
	const Json* md = j.find("model_dict");
	if (!md) throw std::runtime_error("not a VMAF model file");
	Model m;
	for (auto& x : md->find("feature_names")->a) m.names.push_back(x.text());
	for (auto& x : md->find("slopes")->a) m.slopes.push_back(num(x));
	for (auto& x : md->find("intercepts")->a) m.intercepts.push_back(num(x));
	if (auto* c = md->find("score_clip")) { m.clip = true; m.clip_min = num(c->a[0]); m.clip_max = num(c->a[1]); }
	if (auto* t = md->find("score_transform")) {
		if (auto* e = t->find("enabled")) m.transform = truthy(*e);
		auto coef = [&](const char* k, bool& has, double& v) { if (auto* p = t->find(k)) if (p->type == Json::Num) { has = true; v = num(*p); } };
		coef("p0", m.has_p0, m.p0); coef("p1", m.has_p1, m.p1); coef("p2", m.has_p2, m.p2);
		if (auto* k = t->find("knots")) if (!k->a.empty() && m.transform) throw std::runtime_error("models whose score transform has knots are not supported");
		if (auto* v = t->find("out_lte_in")) m.out_lte_in = truthy(*v);
		if (auto* v = t->find("out_gte_in")) m.out_gte_in = truthy(*v);
	}
	if (auto* c = md->find("chroma_correction_parameter")) m.chroma_correction = num(*c);
	if (md->find("norm_type") && md->find("norm_type")->text() != "linear_rescale") throw std::runtime_error("only linear_rescale models are supported");

	// The features and their options (feature_opts_dicts[i] belongs to feature_names[i]).
	const Json* opts = md->find("feature_opts_dicts");
	for (size_t i = 0; i < m.names.size(); i++) {
		const std::string& n = m.names[i];
		const Json* o = opts && i < opts->a.size() ? &opts->a[i] : nullptr;
		auto each = [&](auto f) { if (o) for (auto& kv : o->o) f(kv.first, kv.second); };
		auto bad = [&](const std::string& k) { throw std::runtime_error("the GPU scorer does not know the option " + k + " of " + n); };
		if (n.rfind("VMAF_integer_feature_vif_scale", 0) == 0) {
			m.vif = true;
			each([&](const std::string& k, const Json& v) { if (k == "vif_enhn_gain_limit") m.vif_egl = num(v); else bad(k); });
		} else if (n == "VMAF_integer_feature_adm2_score" || n == "VMAF_integer_feature_adm3_score") {
			m.adm = true;
			if (n == "VMAF_integer_feature_adm3_score") m.aim = true;
			each([&](const std::string& k, const Json& v) {
				AdmOpts& a = m.adm_o;
				if (k == "adm_enhn_gain_limit") a.egl = num(v);
				else if (k == "adm_norm_view_dist") a.nvd = num(v);
				else if (k == "adm_ref_display_height") a.rdh = (int)num(v);
				else if (k == "adm_csf_mode") a.csf_mode = (int)num(v);
				else if (k == "adm_csf_scale") a.csf_scale = num(v);
				else if (k == "adm_csf_diag_scale") a.csf_diag_scale = num(v);
				else if (k == "adm_noise_weight") a.noise_weight = num(v);
				else if (k == "adm_dlm_weight") a.dlm_weight = num(v);
				else if (k == "adm_min_val") a.min_val = num(v);
				else if ((k == "adm_skip_aim" || k == "adm_skip_scale0") && !truthy(v)) {}
				else if (k == "adm_norm_view_dist_extra" && num(v) <= 0) {}
				else bad(k);
			});
		} else if (n == "VMAF_integer_feature_motion2_score" || n == "VMAF_integer_feature_motion3_score") {
			m.motion = true;
			each([&](const std::string& k, const Json& v) {
				MotionOpts& a = m.motion_o;
				if (k == "motion_max_val") a.max_val = num(v);
				else if (k == "motion_blend_factor") a.blend_factor = num(v);
				else if (k == "motion_blend_offset") a.blend_offset = num(v);
				else if (k == "motion_fps_weight") a.fps_weight = num(v);
				else if (k == "motion_five_frame_window") a.five_frame = truthy(v);
				else if (k == "motion_moving_average") a.moving_average = truthy(v);
				else if (k == "motion_force_zero" && !truthy(v)) {}
				else bad(k);
			});
		} else if (n == "Cambi_feature_cambi_score") {
			m.cambi = true;
			each([&](const std::string& k, const Json& v) {
				CambiOpts& a = m.cambi_o;
				if (k == "cambi_high_res_speedup") a.high_res_speedup = (int)num(v);
				else if (k == "cambi_vis_lum_threshold") a.vlt = num(v);
				else if (k == "cambi_max_val") a.max_val = num(v);
				else if (k == "cambi_topk" || k == "topk") a.topk = num(v);
				else if (k == "tvi_threshold") a.tvi = num(v);
				else if (k == "window_size") a.window = (int)num(v);
				else if (k == "max_log_contrast") a.max_log_contrast = (int)num(v);
				else if (k == "eotf" || k == "cambi_eotf") a.eotf = v.text();
				else if (k == "full_ref" && !truthy(v)) {}
				else bad(k);
			});
		} else if (n == "Speed_chroma_feature_speed_chroma_uv_score" || n == "Speed_chroma_feature_speed_chroma_u_score" || n == "Speed_chroma_feature_speed_chroma_v_score") {
			m.speed = true;
			each([&](const std::string& k, const Json& v) {
				SpeedOpts& a = m.speed_o;
				if (k == "speed_kernelscale") a.kernelscale = num(v);
				else if (k == "speed_prescale") a.prescale = num(v);
				else if (k == "speed_prescale_method") a.prescale_method = v.text();
				else if (k == "speed_sigma_nn") a.sigma_nn = num(v);
				else if (k == "speed_nn_floor") a.nn_floor = num(v);
				else if (k == "speed_max_val") a.max_val = num(v);
				else if (k == "speed_weight_var_mode") a.weight_var_mode = (int)num(v);
				else bad(k);
			});
		} else {
			throw std::runtime_error("the model uses a feature the GPU scorer does not have: " + n);
		}
	}
	if (m.adm && m.adm_o.csf_mode == 1) throw std::runtime_error("the Barten CSF (adm_csf_mode 1) is not supported");
	if (m.adm && (m.adm_o.csf_mode < 0 || m.adm_o.csf_mode > 3)) throw std::runtime_error("unknown adm_csf_mode");
	if (m.adm && m.adm_o.csf_mode >= 2 && vmaf_tables::barten_watson_blend_csf(0, 0, m.adm_o.nvd, m.adm_o.rdh) < 0)
		throw std::runtime_error("the blended CSF has no table for this viewing distance and display height");
	if (m.cambi && (m.cambi_o.max_log_contrast < 0 || m.cambi_o.max_log_contrast > 3)) throw std::runtime_error("CAMBI's max_log_contrast must be 0..3");
	if (m.cambi && m.cambi_o.eotf != "bt1886" && m.cambi_o.eotf != "pq") throw std::runtime_error("unknown CAMBI eotf");
	if (m.speed && m.speed_o.prescale_method != "nearest" && m.speed_o.prescale_method != "bilinear") throw std::runtime_error("only nearest and bilinear prescaling are supported");
	if (m.speed && (m.speed_o.weight_var_mode < 0 || m.speed_o.weight_var_mode > 6)) throw std::runtime_error("unknown speed_weight_var_mode");

	std::istringstream in(md->find("model")->text());
	std::string line;
	bool sv = false;
	while (std::getline(in, line)) {
		if (!sv) {
			std::istringstream l(line);
			std::string key;
			l >> key;
			if (key == "gamma") l >> m.gamma;
			else if (key == "rho") l >> m.rho;
			else if (key == "svm_type") { std::string k; l >> k; if (k != "nu_svr") throw std::runtime_error("only nu_svr models are supported"); }
			else if (key == "kernel_type") { std::string k; l >> k; if (k != "rbf") throw std::runtime_error("only the rbf kernel is supported"); }
			else if (key == "SV") sv = true;
			continue;
		}
		if (line.empty()) continue;
		std::istringstream l(line);
		double c;
		l >> c;
		std::vector<double> x(m.names.size(), 0.0);
		std::string tok;
		while (l >> tok) {
			const size_t colon = tok.find(':');
			if (colon == std::string::npos) continue;
			const int idx = atoi(tok.substr(0, colon).c_str());
			if (idx >= 1 && idx <= (int)x.size()) x[idx - 1] = atof(tok.c_str() + colon + 1);
		}
		m.coef.push_back(c);
		m.sv.push_back(std::move(x));
	}
	if (m.slopes.size() != m.names.size() + 1 || m.intercepts.size() != m.names.size() + 1 || m.sv.empty()) throw std::runtime_error("incomplete model");
	return m;
}

// --- ADM: the CSF factors (integer_adm.c) ---
struct dwt_model_params { float a, k, f0, g[4]; };
const dwt_model_params threshold0 = {0.495f, 0.466f, 0.401f, {1.501f, 1.0f, 0.534f, 1.0f}};
const float amplitudes[6][4] = {
	{0.62171, 0.67234, 0.72709, 0.67234}, {0.34537, 0.41317, 0.49428, 0.41317}, {0.18004, 0.22727, 0.28688, 0.22727},
	{0.091401, 0.11792, 0.15214, 0.11792}, {0.045943, 0.059758, 0.077727, 0.059758}, {0.023013, 0.030018, 0.039156, 0.030018}};
float dwt_quant_step(const dwt_model_params* params, int lambda, int theta, double adm_norm_view_dist, int adm_ref_display_height) {
	float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;
	float temp = std::log10(std::pow(2.0, lambda + 1) * params->f0 * params->g[theta] / r);
	float Q = 2.0 * params->a * std::pow(10.0, params->k * temp * temp) / amplitudes[lambda][theta];
	return Q;
}
// factor1 (horizontal and vertical bands) and factor2 (diagonal) of a scale.
void csf_factors(const AdmOpts& o, int scale, float& f1, float& f2) {
	using namespace vmaf_tables;
	if (o.csf_mode == 1) {
		f1 = barten_csf(scale, o.nvd, o.rdh, 100.0, o.csf_scale);
		f2 = barten_csf(scale, o.nvd, o.rdh, 100.0, o.csf_diag_scale);
	} else if (o.csf_mode == 2) {
		f1 = barten_watson_blend_csf(scale, 0, o.nvd, o.rdh);
		f2 = barten_watson_blend_csf(scale, 1, o.nvd, o.rdh);
	} else if (o.csf_mode == 3) {
		f1 = barten_watson_blend_csf_mae(scale, 0, o.nvd, o.rdh);
		f2 = barten_watson_blend_csf_mae(scale, 1, o.nvd, o.rdh);
	} else {
		f1 = 1.0f / dwt_quant_step(&threshold0, scale, 1, o.nvd, o.rdh);
		f2 = 1.0f / dwt_quant_step(&threshold0, scale, 2, o.nvd, o.rdh);
	}
}

struct VifPC { int W, H, scale, srckind, bpc, pad; double egl; };
struct AdmPC {
	int SW, SH, BW, BH, scale, prevN;
	double egl;
	float cos1sq;
	int rf0, rf1, rf2;
	int dl, dt, dr, db;
	int cl, ct, cr, cb;
	int srow, erow, scol, ecol;
	int shc0, shc1, shc2;
	int bpc;
	int aim;
};
struct CambiPC { int W, H, OW, OH, bpc, anti_dither, mask_index, pad, nd, vlt_luma, band_base, band_size; };
struct SpeedPC { int W, H, SW, SH, DW, DH, bpc, fw, scaled, out_off; float scaler; };

// --- CAMBI: luminance (luminance_tools.c) and the thresholds that follow from it (cambi.c) ---
struct LumaRange { int foot, head; };
double bt1886_eotf(double V) {
	const double gamma = 2.4, lw = 300.0, lb = 0.01;
	double a = std::pow(std::pow(lw, 1.0 / gamma) - std::pow(lb, 1.0 / gamma), gamma);
	double b = std::pow(lb, 1.0 / gamma) / (std::pow(lw, 1.0 / gamma) - std::pow(lb, 1.0 / gamma));
	return a * std::pow(std::max(V + b, 0.0), gamma);
}
double pq_eotf(double V) {
	const double m_1 = 0.1593017578125, m_2 = 78.84375, c_1 = 0.8359375, c_2 = 18.8515625, c_3 = 18.6875;
	double num = std::pow(V, 1.0 / m_2) - c_1;
	double num_clipped = std::max(num, 0.0);
	double den = c_2 - c_3 * std::pow(V, 1.0 / m_2);
	return 10000 * std::pow(num_clipped / den, 1.0 / m_1);
}
double luminance(int sample, LumaRange r, bool pq) {
	const int c = sample < r.foot ? r.foot : (sample > r.head ? r.head : sample);
	const double n = (double)(c - r.foot) / (r.head - r.foot);
	return pq ? pq_eotf(n) : bt1886_eotf(n);
}
bool tvi_condition(int sample, int diff, double thr, LumaRange r, bool pq) {
	double mean = luminance(sample, r, pq);
	double d = luminance(sample + diff, r, pq);
	return (d - mean) > thr * mean;
}
int tvi_bisect(int sample, int diff, double thr, LumaRange r, bool pq) { // 0: too small, 1: correct, 2: too big
	if (!tvi_condition(sample, diff, thr, r, pq)) return 2;
	if (tvi_condition(sample + 1, diff, thr, r, pq)) return 0;
	return 1;
}
int tvi_for_diff(int diff, double thr, int bitdepth, LumaRange r, bool pq) {
	const int max_val = (1 << bitdepth) - 1;
	int foot = r.foot, head = r.head - diff - 1;
	int t = tvi_bisect(foot, diff, thr, r, pq);
	if (t == 2) return 0;
	if (t == 1) return foot;
	t = tvi_bisect(head, diff, thr, r, pq);
	if (t == 0) return max_val;
	if (t == 1) return head;
	while (true) {
		int mid = foot + (head - foot) / 2;
		t = tvi_bisect(mid, diff, thr, r, pq);
		if (t == 2) head = mid;
		else if (t == 0) foot = mid;
		else return mid;
	}
}
int vlt_luma(double thr, LumaRange r, bool pq) {
	uint16_t sample = (uint16_t)r.foot;
	while (luminance(sample, r, pq) < thr) sample++;
	return sample == r.foot ? 0 : sample;
}
const int contrast_weights[32] = {1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8};

// The sizes CAMBI works at: the encoded size, then each scale.
struct CambiGeom {
	int ew = 0, eh = 0, window = 0, pad = 0, speedup = 0;
	int sw[5] = {}, sh[5] = {};
	bool ok(std::string& why) const {
		if (ew < 216 && eh < 216) { why = "CAMBI needs frames of at least 216 samples one way"; return false; }
		for (int s = 0; s < 5; s++) if (sw[s] < pad + 1 || sh[s] < pad + 1) { why = "the frame is too small for CAMBI on the GPU"; return false; }
		if (window * window >= CAMBI_RECIPROCAL_LUT_SIZE) { why = "CAMBI's window is too large"; return false; }
		return true;
	}
};
CambiGeom cambi_geom(const CambiOpts& o, int ew, int eh) {
	CambiGeom g;
	g.ew = ew; g.eh = eh;
	const int pix = ew * eh;
	g.speedup = (o.high_res_speedup == 1080 && pix >= 1920 * 1080) || (o.high_res_speedup == 1440 && pix >= 2560 * 1440) || (o.high_res_speedup == 2160 && pix >= 3840 * 2160);
	uint16_t ws = (uint16_t)o.window;
	ws = (uint16_t)(((ws * (ew + eh)) / 375) >> 4);
	if (g.speedup) ws = (uint16_t)((ws + 1) >> 1);
	ws |= 1;
	g.window = ws;
	g.pad = ws >> 1;
	int w = ew, h = eh;
	for (int s = 0; s < 5; s++) {
		if (s > 0 || g.speedup) { w = (w + 1) >> 1; h = (h + 1) >> 1; }
		g.sw[s] = w; g.sh[s] = h;
	}
	return g;
}

// --- SpEED: the sizes (speed.c, speed_init_dimensions()) ---
vmaf_speed::SpeedDimensions speed_dims(int w, int h, double prescale) {
	vmaf_speed::SpeedDimensions d{};
	d.original_height = h; d.original_width = w;
	d.scaled_height = (int)(d.original_height * prescale + 0.5);
	d.scaled_width = (int)(d.original_width * prescale + 0.5);
	d.alloc_height = std::max(d.original_height, d.scaled_height);
	d.alloc_width = std::max(d.original_width, d.scaled_width);
	d.operating_height = d.scaled_height >> 4;
	d.operating_width = d.scaled_width >> 4;
	d.block_size = 5;
	d.truncated_width = (d.operating_width / d.block_size) * d.block_size;
	d.truncated_height = (d.operating_height / d.block_size * d.block_size);
	d.num_blocks_horizontal = d.truncated_width / d.block_size;
	d.num_blocks_vertical = d.truncated_height / d.block_size;
	d.num_blocks = d.num_blocks_horizontal * d.num_blocks_vertical;
	d.elements_in_block = d.block_size * d.block_size;
	d.submatrix_width = d.truncated_width - d.block_size + 1;
	d.submatrix_height = d.truncated_height - d.block_size + 1;
	return d;
}
void chroma_size(int chroma, int w, int h, int& cw, int& ch) {
	cw = chroma == 444 ? w : w / 2;
	ch = chroma == 420 ? h / 2 : h;
}

// One frame's features for one model.
struct Features {
	double adm2 = 0, aim = 0, adm[4] = {}, adm_num[4] = {}, adm_den[4] = {}, vif[4] = {}, sad = 0, cambi = 0, speed_u = 0, speed_v = 0, speed_uv = 0;
};

// What the shaders compute depends only on some of a feature's options; models that agree on those share the work, the rest is applied on the CPU.
std::string adm_gpu_key(const AdmOpts& o) { // decouple, csf and masking; the denominator and the wavelets are the same for all
	char b[256];
	snprintf(b, sizeof b, "%.17g %.17g %d %d %.17g %.17g", o.egl, o.nvd, o.rdh, o.csf_mode, o.csf_scale, o.csf_diag_scale);
	return b;
}
std::string cambi_gpu_key(const CambiOpts& o) {
	char b[256];
	snprintf(b, sizeof b, "%.17g %.17g %d %d %d %s", o.tvi, o.vlt, o.window, o.max_log_contrast, o.high_res_speedup, o.eotf.c_str());
	return b;
}
std::string speed_gpu_key(const SpeedOpts& o) {
	char b[256];
	snprintf(b, sizeof b, "%.17g %.17g %s", o.kernelscale, o.prescale, o.prescale_method.c_str());
	return b;
}
// Several models scored in one pass must agree on CAMBI's options (one CAMBI pass).
bool models_compatible(const std::vector<Model>& ms, std::string& why) {
	const Model* c = nullptr;
	for (auto& m : ms) {
		if (!m.cambi) continue;
		if (c && cambi_gpu_key(c->cambi_o) != cambi_gpu_key(m.cambi_o)) { why = "the models use CAMBI with different options"; return false; }
		c = &m;
	}
	return true;
}

class VkVmaf {
public:
	VkVmaf(int device, const std::vector<Model>& models, const GpuVmafFrame& f) : vk(device), ms(models), W(f.w), H(f.h), bpc(f.depth), bps(f.depth > 8 ? 2 : 1) {
		ysize = (size_t)W * H;
		plane_bytes = ysize * bps;
		raw_size = (plane_bytes + 3) / 4 * 4 + 4;
		raw_ref = vk.buffer(raw_size, 0);
		raw_dis = vk.buffer(raw_size, 0);
		// which model uses which shared part
		for (auto& m : ms) {
			if (m.motion) need_stride[m.motion_o.five_frame ? 1 : 0] = true;
			if (m.vif) { int u = 0; while (u < (int)vif_egl.size() && vif_egl[u] != m.vif_egl) u++; if (u == (int)vif_egl.size()) vif_egl.push_back(m.vif_egl); vif_of.push_back(u); } else vif_of.push_back(-1);
			if (m.adm) {
				const std::string k = adm_gpu_key(m.adm_o);
				int u = 0; while (u < (int)au.size() && au[u].key != k) u++;
				if (u == (int)au.size()) { au.emplace_back(); au[u].key = k; au[u].o = m.adm_o; }
				au[u].aim = au[u].aim || m.aim;
				adm_of.push_back(u);
			} else adm_of.push_back(-1);
			if (m.cambi && !cambi_on) { cambi_on = true; cambi_o = m.cambi_o; }
			if (m.speed) {
				const std::string k = speed_gpu_key(m.speed_o);
				int u = 0; while (u < (int)su.size() && su[u].key != k) u++;
				if (u == (int)su.size()) { su.emplace_back(); su[u].key = k; su[u].o = m.speed_o; }
				speed_of.push_back(u);
			} else speed_of.push_back(-1);
		}
		if (need_stride[0] || need_stride[1]) init_motion();
		if (!vif_egl.empty()) init_vif();
		if (!au.empty()) init_adm();
		if (cambi_on) init_cambi(f);
		if (!su.empty()) init_speed(f);
	}

	// Scores one frame pair, for every model. `index` 0 has no motion.
	std::vector<Features> process(const uint8_t* const ref[3], const size_t ref_stride[3], const uint8_t* const dis[3], const size_t dis_stride[3], size_t index) {
		auto put = [&](void* dst, const uint8_t* src, size_t stride, int w, int h) {
			const size_t row = (size_t)w * bps;
			if (stride == row) memcpy(dst, src, row * h);
			else for (int y = 0; y < h; y++) memcpy((uint8_t*)dst + (size_t)y * row, src + (size_t)y * stride, row);
		};
		put(raw_ref.map, ref[0], ref_stride[0], W, H);
		put(raw_dis.map, dis[0], dis_stride[0], W, H);
		const int cur = (int)(index % 3);
		const bool motion = need_stride[0] || need_stride[1];
		if (motion) put(mfr[cur].map, ref[0], ref_stride[0], W, H);
		if (!su.empty()) {
			put(sraw[0].map, ref[1], ref_stride[1], cw, ch);
			put(sraw[1].map, ref[2], ref_stride[2], cw, ch);
			put(sraw[2].map, dis[1], dis_stride[1], cw, ch);
			put(sraw[3].map, dis[2], dis_stride[2], cw, ch);
		}
		vk.begin();
		bool with_motion[2] = {false, false};
		for (int k = 0; k < 2; k++) {
			const int stride = k + 1;
			if (!need_stride[k] || index < (size_t)stride) continue;
			with_motion[k] = true;
			const int prev = (int)((index - stride) % 3);
			const int pc[3] = {W, H, bpc};
			vk.push(pc, sizeof pc);
			vk.bind(my[k][prev][cur]); vk.dispatch(gx, gy); vk.barrier();
			vk.bind(mx[k][prev][cur]); vk.dispatch(gx, gy); vk.barrier();
		}
		auto groups = [](int a, int b) { return std::make_pair((uint32_t)((a + 15) / 16), (uint32_t)((b + 15) / 16)); };
		if (!vif_egl.empty()) {
			for (int s = 0; s < 4; s++) {
				if (s > 0) {
					VifPC pc{vw[s - 1], vh[s - 1], s, s == 1 ? (bps == 2 ? 1 : 0) : 2, bpc, 0, vif_egl[0]};
					vk.push(&pc, sizeof pc);
					vk.bind(vsub_v[s]);
					auto g = groups(vw[s - 1], vh[s - 1] / 2);
					vk.dispatch(g.first, g.second); vk.barrier();
					vk.bind(vsub_h[s]);
					g = groups(vw[s - 1] / 2, vh[s - 1] / 2);
					vk.dispatch(g.first, g.second); vk.barrier();
				}
				for (size_t u = 0; u < vif_egl.size(); u++) { // the statistics with each enhancement gain limit
					VifPC pc{vw[s], vh[s], s, s == 0 ? (bps == 2 ? 1 : 0) : 2, bpc, 0, vif_egl[u]};
					vk.push(&pc, sizeof pc);
					auto g = groups(vw[s], vh[s]);
					vk.bind(vstat_v[u][s]); vk.dispatch(g.first, g.second); vk.barrier();
					vk.bind(vstat_h[u][s]); vk.dispatch(g.first, g.second); vk.barrier();
				}
			}
		}
		if (!au.empty()) {
			for (int s = 0; s < 4; s++) {
				// the wavelets and the denominator, once
				AdmPC pc = au[0].apc[s];
				vk.push(&pc, sizeof pc);
				vk.bind(adwt_v[s]);
				auto g = groups(sw[s], bh[s]);
				vk.dispatch(g.first, g.second); vk.barrier();
				vk.bind(adwt_h[s]);
				g = groups(bw[s], bh[s]);
				vk.dispatch(g.first, g.second); vk.barrier();
				AdmPC pd = pc;
				if (s > 0) pd.shc0 = au[0].ah[s].den_cub;
				vk.push(&pd, sizeof pd);
				vk.bind(aden_p[s]); vk.dispatch(1, pc.cb - pc.ct);
				// then decouple, csf and masking with each set of options, unit 0 in place, the others on a copy of the bands
				for (size_t u = 0; u < au.size(); u++) {
					AdmUnit& a = au[u];
					if (u > 0) {
						const int n = 8 * bw[s] * bh[s];
						vk.push(&n, sizeof n);
						vk.bind(a.copy[s]);
						const uint32_t total = (uint32_t)((n + 255) / 256), cx = std::min<uint32_t>(total, 65535), cy = (total + cx - 1) / cx;
						vk.dispatch(cx, cy); vk.barrier();
					}
					AdmPC pu = a.apc[s];
					vk.push(&pu, sizeof pu);
					vk.bind(a.dec[s]); vk.dispatch(g.first, g.second); vk.barrier();
					vk.bind(a.csf[s]); vk.dispatch(g.first, g.second); vk.barrier();
					vk.bind(a.cm_p[s]); vk.dispatch(1, pu.erow - pu.srow);
					vk.barrier();
					if (a.aim) { // the csf and the masking again, the other way round, into buffers of their own
						AdmPC pa = pu;
						pa.aim = 1;
						vk.push(&pa, sizeof pa);
						vk.bind(a.aimcsf[s]); vk.dispatch(g.first, g.second); vk.barrier();
						vk.bind(a.aimcm[s]); vk.dispatch(1, pu.erow - pu.srow);
						vk.barrier();
					}
				}
			}
		}
		if (cambi_on) record_cambi();
		for (auto& u : su) {
			for (int p = 0; p < 4; p++) {
				SpeedPC pc = u.spc;
				pc.out_off = p * u.spc.DW * u.spc.DH;
				vk.push(&pc, sizeof pc);
				vk.bind(u.spipe[p]);
				vk.dispatch((uint32_t)((u.spc.DW + 7) / 8), (uint32_t)((u.spc.DH + 7) / 8));
			}
		}
		vk.submit_wait();

		// the sums, once for each shared part
		double sad_raw[2] = {0, 0};
		for (int k = 0; k < 2; k++) {
			if (!with_motion[k]) continue;
			const uint32_t* p = (const uint32_t*)mpart[k].map;
			uint64_t sum = 0;
			for (uint32_t q = 0; q < gx * gy; q++) sum += p[q];
			sad_raw[k] = (double)sum / 256. / ((double)W * H);
		}
		std::vector<std::array<double, 4>> vif(vif_egl.size());
		for (size_t u = 0; u < vif_egl.size(); u++) {
			for (int s = 0; s < 4; s++) {
				const int64_t* p = (const int64_t*)vpart[u][s].map;
				int64_t sum[4] = {0, 0, 0, 0};
				for (uint32_t k = 0; k < vgroups[s]; k++) for (int q = 0; q < 4; q++) sum[q] += p[k * 4 + q];
				const float num = (float)(sum[0] / 2048.0 + (sum[3] - ((sum[2]) / 16384.0) / (65025.0)));
				const float den = (float)(sum[1] / 2048.0 + sum[3]);
				vif[u][s] = (double)(num / den);
			}
		}
		std::map<std::pair<int, double>, Features> adm_done; // by unit and noise weight
		std::map<double, double> cambi_done; // by topk (before the clip to max_val)
		std::map<std::string, std::array<float, 2>> speed_done; // u and v by unit and host-side options
		std::map<std::string, std::array<bool, 2>> speed_err;
		std::vector<Features> out(ms.size());
		for (size_t i = 0; i < ms.size(); i++) {
			const Model& m = ms[i];
			Features& f = out[i];
			if (m.motion) {
				const int k = m.motion_o.five_frame ? 1 : 0;
				if (with_motion[k]) f.sad = std::min(sad_raw[k] * m.motion_o.fps_weight, m.motion_o.max_val);
			}
			if (m.vif) for (int s = 0; s < 4; s++) f.vif[s] = vif[vif_of[i]][s];
			if (m.adm) {
				const auto key = std::make_pair(adm_of[i], m.adm_o.noise_weight);
				auto it = adm_done.find(key);
				if (it == adm_done.end()) { Features a; adm_scores(au[adm_of[i]], m.adm_o.noise_weight, a); it = adm_done.emplace(key, a).first; }
				f.adm2 = it->second.adm2; f.aim = it->second.aim;
				for (int s = 0; s < 4; s++) { f.adm[s] = it->second.adm[s]; f.adm_num[s] = it->second.adm_num[s]; f.adm_den[s] = it->second.adm_den[s]; }
			}
			if (m.cambi) {
				auto it = cambi_done.find(m.cambi_o.topk);
				if (it == cambi_done.end()) it = cambi_done.emplace(m.cambi_o.topk, cambi_score(m.cambi_o.topk)).first;
				f.cambi = std::min(it->second, m.cambi_o.max_val);
			}
			if (m.speed) {
				const SpeedOpts& o = m.speed_o;
				char b[160];
				snprintf(b, sizeof b, "%d %.17g %.17g %d", speed_of[i], o.sigma_nn, o.nn_floor, o.weight_var_mode);
				auto it = speed_done.find(b);
				if (it == speed_done.end()) {
					bool eu, ev;
					const float u = speed_plane(su[speed_of[i]], o, 0, eu), v = speed_plane(su[speed_of[i]], o, 1, ev);
					it = speed_done.emplace(b, std::array<float, 2>{u, v}).first;
					speed_err[b] = {eu, ev};
				}
				const float u = it->second[0], v = it->second[1];
				const bool err_u = speed_err[b][0], err_v = speed_err[b][1];
				float uv;
				if (err_u && !err_v) uv = v;
				else if (err_v && !err_u) uv = u;
				else uv = (u + v) / 2.0;
				const double mx = o.max_val;
				f.speed_u = std::min((double)u, mx); f.speed_v = std::min((double)v, mx); f.speed_uv = std::min((double)uv, mx);
			}
		}
		return out;
	}

private:
	struct AdmHost { int left, top, right, bottom; uint32_t shift_inner; int shift_cub[3]; float rfactor[3]; uint32_t i_rfactor[3]; int den_cub; };
	// ADM with one set of the options the shaders use. Unit 0 works in the buffers the wavelets write; the others on a copy of their bands.
	struct AdmUnit {
		std::string key;
		AdmOpts o;
		bool aim = false;
		AdmPC apc[4];
		AdmHost ah[4];
		Buf cur[4], cm[4], aimb[4];
		Pipe copy[4], dec[4], csf[4], cm_p[4], aimcsf[4], aimcm[4];
	};
	// SpEED with one set of the options the shaders use (the decimation by 16, after an optional prescale).
	struct SpeedUnit {
		std::string key;
		SpeedOpts o;
		vmaf_speed::SpeedDimensions sd{};
		SpeedPC spc{};
		Buf stab, sfilt, sout;
		Pipe spipe[4];
		std::vector<float> sfilter3;
		int sfw3 = 0;
	};
	Vk vk; // first: it releases the buffers and sets below when it goes away
	const std::vector<Model>& ms;
	int W, H;
	int bpc = 8, bps = 1; // bits per sample, bytes per sample
	size_t ysize = 0, plane_bytes = 0, raw_size = 0;
	Buf raw_ref, raw_dis;
	std::vector<int> vif_of, adm_of, speed_of; // per model: its unit, -1 without the feature
	// motion: the last three reference frames; stride 1 compares with the previous frame, stride 2 (the five-frame window) with the one before
	bool need_stride[2] = {false, false};
	uint32_t gx = 0, gy = 0;
	Buf mfr[3], mtmp, mpart[2];
	Pipe my[2][3][3], mx[2][3][3];
	// vif: the subsampled planes once, the statistics for each enhancement gain limit
	std::vector<double> vif_egl;
	int vw[4], vh[4];
	uint32_t vgroups[4];
	Buf vdr[4], vdd[4], vtmp, vlog;
	std::vector<std::array<Buf, 4>> vpart;
	Pipe vsub_v[4], vsub_h[4];
	std::vector<std::array<Pipe, 4>> vstat_v, vstat_h;
	// adm
	int sw[4], sh[4], bw[4], bh[4];
	std::vector<AdmUnit> au;
	Buf acur[4], aden[4], atmp, adiv;
	Pipe adwt_v[4], adwt_h[4], aden_p[4];
	// cambi
	bool cambi_on = false;
	CambiOpts cambi_o;
	CambiGeom cg;
	CambiPC cpc{};
	bool anti_dither = false;
	Buf cidx, cfull, cdith, cmask0, cimg[5], cmask[5], ctmp, ccval[5], cpart[5], ctab;
	Pipe cp_pre, cp_dither, cp_mask, cp_dec[5], cp_mode_h[5], cp_mode_v[5], cp_cval[5];
	uint32_t cgroups[5] = {};
	// speed: the chroma planes once, the decimation per unit
	int cw = 0, ch = 0;
	Buf sraw[4];
	std::vector<SpeedUnit> su;

	void init_motion() {
		gx = (W + 15) / 16; gy = (H + 15) / 16;
		for (auto& b : mfr) b = vk.buffer(raw_size, 0);
		mtmp = vk.buffer(ysize * 4, 1);
		for (int k = 0; k < 2; k++) {
			if (!need_stride[k]) continue;
			mpart[k] = vk.buffer((size_t)gx * gy * 4, 2);
			for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) {
				if (a == b) continue;
				std::vector<Buf*> bind{&mfr[a], &mfr[b], &mtmp, &mpart[k]};
				my[k][a][b] = vk.pipeline("motion_y", bind);
				mx[k][a][b] = vk.pipeline("motion_x", bind);
			}
		}
	}

	void init_vif() {
		for (int s = 0; s < 4; s++) { vw[s] = s ? vw[s - 1] / 2 : W; vh[s] = s ? vh[s - 1] / 2 : H; }
		for (int s = 1; s < 4; s++) { vdr[s] = vk.buffer((size_t)vw[s] * vh[s] * 4 + 16, 1); vdd[s] = vk.buffer((size_t)vw[s] * vh[s] * 4 + 16, 1); }
		vpart.resize(vif_egl.size());
		vstat_v.resize(vif_egl.size()); vstat_h.resize(vif_egl.size());
		for (int s = 0; s < 4; s++) {
			vgroups[s] = ((vw[s] + 15) / 16) * ((vh[s] + 15) / 16);
			for (auto& p : vpart) p[s] = vk.buffer((size_t)vgroups[s] * 4 * 8, 2);
		}
		vtmp = vk.buffer(ysize * 5 * 4 + 16, 1);
		vlog = vk.buffer(65536 * 4, 0);
		{
			uint32_t* t = (uint32_t*)vlog.map;
			memset(t, 0, 65536 * 4);
			for (unsigned i = 32767; i < 65536; ++i) t[i] = (uint16_t)round(log2f((float)i) * 2048);
		}
		auto sref = [&](int s) -> Buf* { return s == 0 ? &raw_ref : &vdr[s]; };
		auto sdis = [&](int s) -> Buf* { return s == 0 ? &raw_dis : &vdd[s]; };
		for (int s = 1; s < 4; s++) {
			std::vector<Buf*> b{sref(s - 1), sdis(s - 1), &vtmp, &vlog, &vpart[0][s], &vdr[s], &vdd[s]};
			vsub_v[s] = vk.pipeline("vif_sub_v", b);
			vsub_h[s] = vk.pipeline("vif_sub_h", b);
		}
		for (size_t u = 0; u < vif_egl.size(); u++) {
			for (int s = 0; s < 4; s++) {
				std::vector<Buf*> b{sref(s), sdis(s), &vtmp, &vlog, &vpart[u][s], &vtmp, &vtmp};
				vstat_v[u][s] = vk.pipeline("vif_stat_v", b);
				vstat_h[u][s] = vk.pipeline("vif_stat_h", b);
			}
		}
	}

	// The geometry of the scales, then each unit's csf factors (integer_adm.c, adm_init / integer_compute_adm).
	void init_adm() {
		for (int s = 0; s < 4; s++) {
			sw[s] = s == 0 ? W : bw[s - 1]; sh[s] = s == 0 ? H : bh[s - 1];
			bw[s] = (sw[s] + 1) / 2; bh[s] = (sh[s] + 1) / 2;
		}
		const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
		for (AdmUnit& a : au) {
			const AdmOpts& o = a.o;
			for (int s = 0; s < 4; s++) {
				AdmPC& pc = a.apc[s];
				memset(&pc, 0, sizeof pc);
				AdmHost& hs = a.ah[s];
				pc.SW = sw[s]; pc.SH = sh[s]; pc.BW = bw[s]; pc.BH = bh[s]; pc.scale = s; pc.bpc = bpc;
				pc.prevN = s ? bw[s - 1] * bh[s - 1] : 0;
				pc.egl = o.egl; pc.cos1sq = cos_1deg_sq;
				const int w = bw[s], h = bh[s];
				int dl = w * 0.1 - 0.5 - 1, dt = h * 0.1 - 0.5 - 1, dr = w - dl + 2, db = h - dt + 2;
				if (dl < 0) dl = 0;
				if (dr > w) dr = w;
				if (dt < 0) dt = 0;
				if (db > h) db = h;
				pc.dl = dl; pc.dt = dt; pc.dr = dr; pc.db = db;
				hs.left = w * 0.1 - 0.5; hs.top = h * 0.1 - 0.5; hs.right = w - hs.left; hs.bottom = h - hs.top;
				if (hs.left <= 0 || hs.top <= 0) throw std::runtime_error("the frame is too small for ADM on the GPU");
				pc.cl = hs.left; pc.ct = hs.top; pc.cr = hs.right; pc.cb = hs.bottom;
				pc.scol = hs.left > 1 ? hs.left : 1; pc.ecol = hs.right < w - 1 ? hs.right : w - 1;
				pc.srow = hs.top > 1 ? hs.top : 1; pc.erow = hs.bottom < h - 1 ? hs.bottom : h - 1;
				float factor1, factor2;
				csf_factors(o, s, factor1, factor2);
				hs.rfactor[0] = factor1; hs.rfactor[1] = factor1; hs.rfactor[2] = factor2;
				if (s == 0) {
					if (std::fabs(o.nvd * o.rdh - 3.0 * 1080) < 1.0e-8 && o.csf_mode == 0) {
						hs.i_rfactor[0] = 36453; hs.i_rfactor[1] = 36453; hs.i_rfactor[2] = 49417;
					} else {
						const double pow2_21 = pow(2, 21), pow2_23 = pow(2, 23);
						hs.i_rfactor[0] = (uint16_t)(hs.rfactor[0] * pow2_21);
						hs.i_rfactor[1] = (uint16_t)(hs.rfactor[1] * pow2_21);
						hs.i_rfactor[2] = (uint16_t)(hs.rfactor[2] * pow2_23);
					}
					hs.shift_cub[0] = (uint32_t)ceil(log2(w) - 4); hs.shift_cub[1] = (uint32_t)ceil(log2(w) - 4); hs.shift_cub[2] = (uint32_t)ceil(log2(w) - 3);
				} else {
					const double pow2_32 = pow(2, 32);
					for (int k = 0; k < 3; k++) hs.i_rfactor[k] = (uint32_t)(hs.rfactor[k] * pow2_32);
					hs.shift_cub[0] = hs.shift_cub[1] = hs.shift_cub[2] = (uint32_t)ceil(log2(w));
				}
				hs.shift_inner = (uint32_t)ceil(log2(h));
				pc.rf0 = (int)hs.i_rfactor[0]; pc.rf1 = (int)hs.i_rfactor[1]; pc.rf2 = (int)hs.i_rfactor[2];
				pc.shc0 = hs.shift_cub[0]; pc.shc1 = hs.shift_cub[1]; pc.shc2 = hs.shift_cub[2];
				hs.den_cub = s ? (int)ceil(log2(hs.right - hs.left)) : 0;
			}
		}
		for (int s = 0; s < 4; s++) {
			const size_t N = (size_t)bw[s] * bh[s];
			acur[s] = vk.buffer(20 * N * 4 + 16, 1);
			aden[s] = vk.buffer((size_t)bh[s] * 3 * 8 + 16, 2);
			for (size_t u = 0; u < au.size(); u++) {
				AdmUnit& a = au[u];
				a.cur[s] = u == 0 ? acur[s] : vk.buffer(20 * N * 4 + 16, 1);
				a.cm[s] = vk.buffer((size_t)bh[s] * 3 * 8 + 16, 2);
				if (a.aim) a.aimb[s] = vk.buffer((size_t)bh[s] * 3 * 8 + 16, 2);
			}
		}
		atmp = vk.buffer((size_t)4 * bh[0] * W * 4 + 16, 1);
		adiv = vk.buffer(65537 * 4, 0);
		{
			int32_t* d = (int32_t*)adiv.map;
			memset(d, 0, 65537 * 4);
			for (int i = 1; i <= 32768; ++i) { const int32_t recip = (int32_t)(1073741824 / i); d[32768 + i] = recip; d[32768 - i] = 0 - recip; }
		}
		for (int s = 0; s < 4; s++) {
			Buf* prev = &acur[s ? s - 1 : 0];
			std::vector<Buf*> b{&acur[s], prev, &raw_ref, &raw_dis, &atmp, &adiv, &au[0].cm[s], &aden[s]};
			adwt_v[s] = vk.pipeline("adm_dwt_v", b);
			adwt_h[s] = vk.pipeline("adm_dwt_h", b);
			aden_p[s] = vk.pipeline("adm_den", b);
			for (size_t u = 0; u < au.size(); u++) {
				AdmUnit& a = au[u];
				if (u > 0) a.copy[s] = vk.pipeline("copy_u32", {&acur[s], &a.cur[s]});
				std::vector<Buf*> bu{&a.cur[s], prev, &raw_ref, &raw_dis, &atmp, &adiv, &a.cm[s], &aden[s]};
				a.dec[s] = vk.pipeline("adm_decouple", bu);
				a.csf[s] = vk.pipeline("adm_csf", bu);
				a.cm_p[s] = vk.pipeline("adm_cm", bu);
				if (a.aim) {
					std::vector<Buf*> ba{&a.cur[s], prev, &raw_ref, &raw_dis, &atmp, &adiv, &a.aimb[s], &aden[s]};
					a.aimcsf[s] = vk.pipeline("adm_csf", ba);
					a.aimcm[s] = vk.pipeline("adm_cm", ba);
				}
			}
		}
	}

	// The numerator of one scale from the masking sums the GPU wrote (integer_adm.c, adm_cm() / i4_adm_cm()).
	float adm_num(const AdmUnit& a, int s, const int64_t* cm, double noise_weight) const {
		const AdmHost& hs = a.ah[s];
		int64_t accum[3] = {0, 0, 0};
		const int rows = a.apc[s].erow - a.apc[s].srow;
		const int64_t add_inner = (int64_t)pow(2, (hs.shift_inner - 1));
		for (int r = 0; r < rows; r++) for (int q = 0; q < 3; q++) accum[q] += (cm[r * 3 + q] + add_inner) >> hs.shift_inner;
		float f_accum[3];
		if (s == 0) {
			f_accum[0] = (float)(accum[0] / pow(2, (52 - hs.shift_cub[0] - hs.shift_inner)));
			f_accum[1] = (float)(accum[1] / pow(2, (52 - hs.shift_cub[1] - hs.shift_inner)));
			f_accum[2] = (float)(accum[2] / pow(2, (57 - hs.shift_cub[2] - hs.shift_inner)));
		} else {
			float final_shift[3] = {(float)pow(2, (45 - hs.shift_cub[0] - hs.shift_inner)), (float)pow(2, (39 - hs.shift_cub[0] - hs.shift_inner)), (float)pow(2, (36 - hs.shift_cub[0] - hs.shift_inner))};
			for (int q = 0; q < 3; q++) f_accum[q] = (float)(accum[q] / final_shift[s - 1]);
		}
		const float add = powf((hs.bottom - hs.top) * (hs.right - hs.left) * noise_weight, 1.0f / 3.0f);
		float parts[3];
		for (int q = 0; q < 3; q++) parts[q] = powf(f_accum[q], 1.0f / 3.0f) + add;
		return (parts[0] + parts[1]) + parts[2];
	}

	void adm_scores(const AdmUnit& a, double noise_weight, Features& f) const {
		double num = 0, den = 0, aim_num = 0, num_s[4], den_s[4];
		for (int s = 0; s < 4; s++) {
			const AdmHost& hs = a.ah[s];
			const int64_t* dn = (const int64_t*)aden[s].map;
			const float num_scale = adm_num(a, s, (const int64_t*)a.cm[s].map, noise_weight);
			const float aim_scale = a.aim ? adm_num(a, s, (const int64_t*)a.aimb[s].map, 0.0) : 0.0f;
			uint64_t acc_d[3] = {0, 0, 0};
			int32_t shift_accum, shift_cub = 0;
			if (s == 0) {
				shift_accum = (int32_t)ceil(log2((hs.bottom - hs.top) * (hs.right - hs.left)) - 20);
				shift_accum = shift_accum > 0 ? shift_accum : 0;
			} else {
				shift_cub = (int32_t)ceil(log2(hs.right - hs.left));
				shift_accum = (int32_t)ceil(log2(hs.bottom - hs.top));
			}
			const int32_t add_shift_accum = (s == 0 ? shift_accum > 0 : true) ? (1 << (shift_accum - 1)) : 0;
			const int drows = hs.bottom - hs.top;
			for (int r = 0; r < drows; r++) for (int q = 0; q < 3; q++) acc_d[q] += ((uint64_t)dn[r * 3 + q] + add_shift_accum) >> shift_accum;
			double shift_csf;
			if (s == 0) shift_csf = pow(2, (18 - shift_accum));
			else { static const uint32_t conv[3] = {32, 27, 23}; shift_csf = pow(2, (conv[s - 1] - shift_accum - shift_cub)); }
			double csf[3];
			for (int q = 0; q < 3; q++) csf[q] = (double)(acc_d[q] / shift_csf) * pow(hs.rfactor[q], 3);
			const float powf_add = powf((hs.bottom - hs.top) * (hs.right - hs.left) * noise_weight, 1.0f / 3.0f);
			float dparts[3];
			for (int q = 0; q < 3; q++) dparts[q] = powf(csf[q], 1.0f / 3.0f) + powf_add;
			const float den_scale = (dparts[0] + dparts[1]) + dparts[2];
			num_s[s] = num_scale; den_s[s] = den_scale;
			num += num_scale; den += den_scale; aim_num += aim_scale;
		}
		const double numden_limit = 1e-10 * (W * H) / (1920.0 * 1080.0);
		const double n = num < numden_limit ? 0 : num, d = den < numden_limit ? 0 : den;
		if (d == 0.0) { f.adm2 = 1.0; f.aim = 0.0; }
		else { f.adm2 = n / d; f.aim = aim_num / d; }
		for (int s = 0; s < 4; s++) { f.adm[s] = num_s[s] / den_s[s]; f.adm_num[s] = num_s[s]; f.adm_den[s] = den_s[s]; }
	}

	void init_cambi(const GpuVmafFrame& f) {
		const CambiOpts& o = cambi_o;
		cg = cambi_geom(o, f.enc_w, f.enc_h);
		std::string why;
		if (!cg.ok(why)) throw std::runtime_error(why);
		anti_dither = f.enc_depth < 10;
		const int ew = cg.ew, eh = cg.eh;
		const size_t full = (size_t)ew * eh;
		// libvmaf's nearest neighbour positions, worked out in float as it does (decimate_generic_*_and_convert_to_10b)
		cidx = vk.buffer((size_t)(ew + eh) * 4 + 16, 0);
		{
			int32_t* idx = (int32_t*)cidx.map;
			if (ew == W && eh == H) {
				for (int j = 0; j < ew; j++) idx[j] = j;
				for (int i = 0; i < eh; i++) idx[ew + i] = i;
			} else {
				float ratio_x = (float)W / ew, ratio_y = (float)H / eh;
				float start_x = ratio_x / 2 - 0.5, start_y = ratio_y / 2 - 0.5;
				float x = start_x;
				for (int j = 0; j < ew; j++) { idx[j] = (int)(x + 0.5); x += ratio_x; }
				float y = start_y;
				for (int i = 0; i < eh; i++) { idx[ew + i] = (int)(y + 0.5); y += ratio_y; }
			}
		}
		cfull = vk.buffer(full * 4 + 16, 1);
		if (anti_dither) cdith = vk.buffer(full * 4 + 16, 1);
		cmask0 = vk.buffer(full * 4 + 16, 1);
		Buf* src = anti_dither ? &cdith : &cfull;
		for (int s = 0; s < 5; s++) {
			const size_t n = (size_t)cg.sw[s] * cg.sh[s];
			if (s == 0 && !cg.speedup) { cimg[0] = *src; cmask[0] = cmask0; }
			else { cimg[s] = vk.buffer(n * 4 + 16, 1); cmask[s] = vk.buffer(n * 4 + 16, 1); }
			ccval[s] = vk.buffer(n * 4 + 16, 2);
			cgroups[s] = (uint32_t)(((cg.sw[s] + 15) / 16) * ((cg.sh[s] + 15) / 16));
			cpart[s] = vk.buffer((size_t)cgroups[s] * 16 + 16, 2);
		}
		ctmp = vk.buffer((size_t)cg.sw[0] * cg.sh[0] * 4 + 16, 1);
		// tvi[32], weight[32], then the reciprocal table
		ctab = vk.buffer((64 + CAMBI_RECIPROCAL_LUT_SIZE) * 4, 0);
		const int nd = 1 << o.max_log_contrast;
		const LumaRange lr{16 * 4, 235 * 4};
		const bool pq = o.eotf == "pq";
		{
			int32_t* t = (int32_t*)ctab.map;
			memset(t, 0, 64 * 4);
			for (int d = 0; d < nd; d++) { t[d] = tvi_for_diff(d + 1, o.tvi, 10, lr, pq) + nd; t[32 + d] = contrast_weights[d]; }
			memcpy(t + 64, vmaf_tables::reciprocal_lut, CAMBI_RECIPROCAL_LUT_SIZE * 4);
			cpc.nd = nd;
			cpc.vlt_luma = vlt_luma(o.vlt, lr, pq);
			const int v_lo = cpc.vlt_luma - 3 * nd + 1;
			cpc.band_base = v_lo > 0 ? v_lo : 0;
			cpc.band_size = (uint16_t)(t[nd - 1] + 1 - cpc.band_base);
		}
		{ // get_mask_index()
			const uint32_t shifted_wh = (uint32_t)(ew >> 6) * (uint32_t)(eh >> 6);
			uint16_t lg = 0;
			if (shifted_wh) { uint32_t tmp = shifted_wh - 1; while (tmp > 0) { tmp >>= 1; lg++; } }
			cpc.mask_index = (uint16_t)((7 * 7 + 3 * (lg - 11) - 1) >> 1);
		}
		cpc.pad = cg.pad;
		cpc.bpc = bpc;
		cpc.anti_dither = anti_dither ? 1 : 0;
		cp_pre = vk.pipeline("cambi_pre", {&raw_dis, &cidx, &cfull});
		if (anti_dither) cp_dither = vk.pipeline("cambi_dither", {&cfull, &cdith});
		cp_mask = vk.pipeline("cambi_mask", {src, &cmask0});
		for (int s = 0; s < 5; s++) {
			if (s > 0 || cg.speedup) {
				Buf* pi = s == 0 ? src : &cimg[s - 1];
				Buf* pm = s == 0 ? &cmask0 : &cmask[s - 1];
				cp_dec[s] = vk.pipeline("cambi_decimate", {pi, pm, &cimg[s], &cmask[s]});
			}
			cp_mode_h[s] = vk.pipeline("cambi_mode", {&cimg[s], &ctmp});
			cp_mode_v[s] = cp_mode_h[s];
			cp_cval[s] = vk.pipeline("cambi_cvalue", {&cimg[s], &cmask[s], &ccval[s], &ctab, &cpart[s]});
		}
	}

	void record_cambi() {
		auto groups = [](int a, int b) { return std::make_pair((uint32_t)((a + 15) / 16), (uint32_t)((b + 15) / 16)); };
		const int ew = cg.ew, eh = cg.eh;
		CambiPC pc = cpc;
		pc.W = W; pc.H = H; pc.OW = ew; pc.OH = eh;
		vk.push(&pc, sizeof pc);
		auto g = groups(ew, eh);
		vk.bind(cp_pre); vk.dispatch(g.first, g.second); vk.barrier();
		pc.W = ew; pc.H = eh;
		vk.push(&pc, sizeof pc);
		if (anti_dither) { vk.bind(cp_dither); vk.dispatch(g.first, g.second); vk.barrier(); }
		vk.bind(cp_mask); vk.dispatch(g.first, g.second); vk.barrier();
		int pw = ew, ph = eh;
		for (int s = 0; s < 5; s++) {
			const int w = cg.sw[s], h = cg.sh[s];
			g = groups(w, h);
			if (s > 0 || cg.speedup) {
				pc.W = pw; pc.H = ph; pc.OW = w; pc.OH = h;
				vk.push(&pc, sizeof pc);
				vk.bind(cp_dec[s]); vk.dispatch(g.first, g.second); vk.barrier();
			}
			pc.W = w; pc.H = h; pc.OW = w; pc.OH = h;
			pc.anti_dither = 0; // cambi_mode: the pass along the rows
			vk.push(&pc, sizeof pc);
			vk.bind(cp_mode_h[s]); vk.dispatch(g.first, g.second); vk.barrier();
			pc.anti_dither = 1; // ... and along the columns
			vk.push(&pc, sizeof pc);
			vk.bind(cp_mode_v[s]); vk.dispatch(g.first, g.second); vk.barrier();
			pc.anti_dither = cpc.anti_dither;
			vk.push(&pc, sizeof pc);
			vk.bind(cp_cval[s]); vk.dispatch(g.first, g.second); vk.barrier();
			pw = w; ph = h;
		}
	}

	// cambi.c, cambi_score(): the mean of the largest topk of the c values of each scale, weighted by scale.
	double cambi_score(double topk) const { // before the clip to cambi_max_val, which each model does
		static const int scale_weights[5] = {16, 8, 4, 2, 1};
		double score = 0;
		for (int s = 0; s < 5; s++) {
			const int n = cg.sw[s] * cg.sh[s];
			int k = (int)(topk * n);
			k = k < 1 ? 1 : (k > n ? n : k);
			const double* p = (const double*)cpart[s].map;
			double nonzero = 0, sum = 0;
			for (uint32_t g = 0; g < cgroups[s]; g++) { nonzero += p[2 * g]; sum += p[2 * g + 1]; }
			double v;
			if (nonzero <= k) v = sum / k; // the top k are all the values that are not 0, and zeros
			else {
				std::vector<float> c((const float*)ccval[s].map, (const float*)ccval[s].map + n);
				std::nth_element(c.begin(), c.begin() + (k - 1), c.end(), std::greater<float>());
				double t = 0;
				for (int i = 0; i < k; i++) t += c[i];
				v = t / k;
			}
			score += v * scale_weights[s];
		}
		const int pixels = (2 * (cg.window >> 1) + 1) * (2 * (cg.window >> 1) + 1);
		return score / pixels;
	}

	void init_speed(const GpuVmafFrame& f) {
		chroma_size(f.chroma, W, H, cw, ch);
		const size_t craw = ((size_t)cw * ch * bps + 3) / 4 * 4 + 4;
		for (auto& b : sraw) b = vk.buffer(craw, 0);
		for (SpeedUnit& u : su) {
			const SpeedOpts& o = u.o;
			u.sd = speed_dims(cw, ch, o.prescale);
			if (u.sd.truncated_height == 0 || u.sd.truncated_width == 0) throw std::runtime_error("the chroma planes are too small for SpEED");
			const int fw = vmaf_speed::vif_get_filter_size(1, (float)o.kernelscale);
			if (fw > 64) throw std::runtime_error("speed_kernelscale is too large");
			std::vector<float> filt(128);
			vmaf_speed::speed_get_antialias_filter(filt.data(), 4, (float)o.kernelscale);
			u.sfw3 = vmaf_speed::vif_get_filter_size(4, (float)o.kernelscale);
			u.sfilter3.assign(128, 0.0f);
			vmaf_speed::vif_get_filter(u.sfilter3.data(), 4, (float)o.kernelscale);
			const int SW = (int)u.sd.scaled_width, SH = (int)u.sd.scaled_height;
			const bool scaled = !(SW == cw && SH == ch) && !(std::fabs(o.prescale - 1.0) < 1.0e-3);
			u.stab = vk.buffer((size_t)(2 * SW + 2 * SH) * 4 * 2 + 16, 0);
			{
				int32_t* t = (int32_t*)u.stab.map;
				float* fx = (float*)(t + 2 * SW + 2 * SH);
				float* fy = fx + SW;
				auto mirror = [](float i, float left, float right) { return (i < left ? -i : i > right ? 2 * right - i : i); };
				if (o.prescale_method == "bilinear") { // vif_scale_frame_bilinear_precompute_columns_s() and its rows
					float ratio_x = (float)cw / SW;
					for (int x = 0; x < SW; x++) {
						float xx = (x + 0.5) * ratio_x - 0.5;
						t[x] = (int)mirror(floor(xx), 0, cw - 1);
						t[SW + x] = (int)mirror(ceil(xx), 0, cw - 1);
						fx[x] = xx - t[x];
					}
					float ratio_y = (float)ch / SH;
					for (int y = 0; y < SH; y++) {
						float yy = (y + 0.5) * ratio_y - 0.5;
						t[2 * SW + y] = (int)mirror(floor(yy), 0, ch - 1);
						t[2 * SW + SH + y] = (int)mirror(ceil(yy), 0, ch - 1);
						fy[y] = yy - t[2 * SW + y];
					}
				} else { // nearest
					float ratio_x = (float)cw / SW, ratio_y = (float)ch / SH;
					for (int x = 0; x < SW; x++) { t[x] = t[SW + x] = (int)(x * ratio_x); fx[x] = 0; }
					for (int y = 0; y < SH; y++) { t[2 * SW + y] = t[2 * SW + SH + y] = (int)(y * ratio_y); fy[y] = 0; }
				}
			}
			u.sfilt = vk.buffer(128 * 4, 0);
			memcpy(u.sfilt.map, filt.data(), 128 * 4);
			u.spc.W = cw; u.spc.H = ch; u.spc.SW = SW; u.spc.SH = SH;
			u.spc.DW = SW / 16; u.spc.DH = SH / 16;
			u.spc.bpc = bpc; u.spc.fw = fw; u.spc.scaled = scaled ? 1 : 0;
			u.spc.scaler = bpc == 10 ? 4.0f : 1.0f;
			u.sout = vk.buffer((size_t)4 * u.spc.DW * u.spc.DH * 4 + 16, 2);
			for (int p = 0; p < 4; p++) u.spipe[p] = vk.pipeline("speed_dec", {&sraw[p], &u.stab, &u.sfilt, &u.sout});
		}
	}

	// speed.c from filter_and_downscale()'s 3x3 blur on: the score of one plane pair (ref p, dis p + 2); `err` if a covariance matrix was singular.
	float speed_plane(const SpeedUnit& u, const SpeedOpts& o, int p, bool& err) const {
		using namespace vmaf_speed;
		const auto& sd = u.sd;
		const int dw = (int)sd.operating_width, dh = (int)sd.operating_height;
		const size_t E = sd.elements_in_block, nb = sd.num_blocks;
		std::vector<float> independent(nb * E), solution(nb * E), cov(E * E), eig(E), tmp(5 * E * E + E * nb + 8 * E + 64);
		std::vector<float> ent_ref(nb), var_ref(nb), ent_dis(nb), var_dis(nb);
		SpeedState s{};
		s.dimensions = sd;
		s.float_stride = (size_t)dw * sizeof(float);
		s.buffers.independent_term = independent.data();
		s.buffers.linear_system_sol = solution.data();
		s.buffers.cov_mat = cov.data();
		s.buffers.eigenvalues = eig.data();
		s.buffers.tmp_buffer = tmp.data();
		s.compute_cov_kernel = compute_cov_kernel_scalar;
		s.ref_results.entropies = ent_ref.data(); s.ref_results.variances = var_ref.data();
		s.dis_results.entropies = ent_dis.data(); s.dis_results.variances = var_dis.data();
		const float* out = (const float*)u.sout.map;
		std::vector<float> frame((size_t)dw * dh), blur((size_t)dw * dh), row(dw);
		auto prepare = [&](int plane) {
			memcpy(frame.data(), out + (size_t)plane * dw * dh, frame.size() * 4);
			vif_filter1d_s(u.sfilter3.data(), frame.data(), blur.data(), row.data(), dw, dh, dw * 4, dw * 4, u.sfw3);
			subtract_image(frame.data(), blur.data(), dw, dh, (size_t)dw * 4);
		};
		prepare(p);
		const int err_ref = est_params(&s, frame.data(), (float)o.sigma_nn, &s.ref_results);
		prepare(p + 2);
		const int err_dis = est_params(&s, frame.data(), (float)o.sigma_nn, &s.dis_results);
		err = err_ref || err_dis;
		if ((err_ref && !err_dis) || (!err_ref && err_dis)) return 0.0f;
		return get_speed_score(s.dimensions, s.ref_results, s.dis_results, (float)o.sigma_nn, (float)o.nn_floor, o.weight_var_mode);
	}
};

} // namespace

struct GpuVmaf::Impl {
	std::vector<Model> models;
	std::unique_ptr<VkVmaf> gpu;
	std::vector<std::vector<Features>> feats; // per frame, per model
};

GpuVmaf::GpuVmaf() : impl(new Impl) {}
GpuVmaf::~GpuVmaf() { delete impl; }

bool GpuVmaf::supported(const std::vector<std::string>& model_paths, int device, const GpuVmafFrame& f, std::string& why) {
	if (f.depth != 8 && f.depth != 10) { why = "only 8 and 10 bit input is supported"; return false; }
	if (f.w < 160 || f.h < 90) { why = "the frame is too small"; return false; }
	if (model_paths.empty()) { why = "no model"; return false; }
	std::vector<Model> ms;
	for (auto& path : model_paths) {
		try {
			ms.push_back(load_model(path));
		} catch (const std::exception& e) { why = e.what(); return false; }
		const Model& m = ms.back();
		if (m.speed) {
			if (f.chroma != 420 && f.chroma != 422 && f.chroma != 444) { why = "the model needs the chroma planes"; return false; }
			int cw, ch;
			chroma_size(f.chroma, f.w, f.h, cw, ch);
			const auto d = speed_dims(cw, ch, m.speed_o.prescale);
			if (d.truncated_width == 0 || d.truncated_height == 0) { why = "the chroma planes are too small for SpEED"; return false; }
		}
		if (m.cambi && !cambi_geom(m.cambi_o, f.enc_w, f.enc_h).ok(why)) return false;
	}
	if (!models_compatible(ms, why)) return false;
	if (!Vk::available(device, why)) return false;
	return true;
}

std::unique_ptr<GpuVmaf> GpuVmaf::create(const std::vector<std::string>& model_paths, int device, const GpuVmafFrame& f, std::string& why) {
	if (!supported(model_paths, device, f, why)) return nullptr;
	std::unique_ptr<GpuVmaf> g(new GpuVmaf);
	try {
		for (auto& path : model_paths) g->impl->models.push_back(load_model(path));
		g->impl->gpu = std::make_unique<VkVmaf>(device, g->impl->models, f);
	} catch (const std::exception& e) {
		why = e.what();
		return nullptr;
	}
	return g;
}

bool GpuVmaf::add(const uint8_t* const ref[3], const size_t ref_stride[3], const uint8_t* const dis[3], const size_t dis_stride[3]) {
	try {
		impl->feats.push_back(impl->gpu->process(ref, ref_stride, dis, dis_stride, impl->feats.size()));
	} catch (const std::exception& e) {
		fprintf(stderr, "vmaf (GPU): %s\n", e.what());
		return false;
	}
	return true;
}

bool GpuVmaf::finish(std::vector<std::vector<double>>& scores, std::vector<std::vector<std::map<std::string, double>>>* features) {
	const size_t n = impl->feats.size(), nm = impl->models.size();
	scores.assign(nm, std::vector<double>(n, 0.0));
	if (features) features->assign(nm, std::vector<std::map<std::string, double>>(n));
	for (size_t mi = 0; mi < nm; mi++) {
	const Model& m = impl->models[mi];
	auto feat = [&](size_t i) -> const Features& { return impl->feats[i][mi]; };
	// motion2 and motion3 (integer_motion.c, flush())
	const MotionOpts& mo = m.motion_o;
	const unsigned stride = mo.five_frame ? 2 : 1, min_idx = mo.five_frame ? 2 : 1;
	auto blend = [&](double v) { return v * mo.blend_factor + (1 - mo.blend_factor) * std::min(mo.blend_offset, v); };
	double stamp = 0.;
	if (n > min_idx) stamp = std::min(blend(feat(min_idx).sad), mo.max_val);
	double prev_processed = 0.;
	for (size_t i = 0; i < n; i++) {
		const Features& f = feat(i);
		double motion2 = 0, motion3;
		if (i >= min_idx) {
			const int lo = (int)i - (int)(stride - 1), hi = (int)i + 1;
			if (hi >= (int)n) motion2 = f.sad;
			else if (lo >= (int)min_idx) motion2 = std::min(feat(lo).sad, feat(hi).sad);
			else motion2 = feat(hi).sad;
		}
		if (i < min_idx) { motion3 = stamp; prev_processed = stamp; }
		else {
			const double processed = std::min(blend(motion2), mo.max_val);
			motion3 = mo.moving_average ? (processed + prev_processed) / 2.0 : processed;
			prev_processed = processed;
		}
		const double adm3 = std::max(f.adm2 * m.adm_o.dlm_weight + (1 - f.aim) * (1 - m.adm_o.dlm_weight), m.adm_o.min_val);
		const std::map<std::string, double> by_name = {
			{"VMAF_integer_feature_adm2_score", f.adm2}, {"VMAF_integer_feature_adm3_score", adm3}, {"VMAF_integer_feature_aim_score", f.aim},
			{"VMAF_integer_feature_adm_scale0_score", f.adm[0]}, {"VMAF_integer_feature_adm_scale1_score", f.adm[1]},
			{"VMAF_integer_feature_adm_scale2_score", f.adm[2]}, {"VMAF_integer_feature_adm_scale3_score", f.adm[3]},
			{"VMAF_integer_feature_motion_sad_score", f.sad}, {"VMAF_integer_feature_motion2_score", motion2}, {"VMAF_integer_feature_motion3_score", motion3},
			{"VMAF_integer_feature_vif_scale0_score", f.vif[0]}, {"VMAF_integer_feature_vif_scale1_score", f.vif[1]},
			{"VMAF_integer_feature_vif_scale2_score", f.vif[2]}, {"VMAF_integer_feature_vif_scale3_score", f.vif[3]},
			{"Cambi_feature_cambi_score", f.cambi},
			{"Speed_chroma_feature_speed_chroma_u_score", f.speed_u}, {"Speed_chroma_feature_speed_chroma_v_score", f.speed_v},
			{"Speed_chroma_feature_speed_chroma_uv_score", f.speed_uv}};
		std::vector<double> x;
		for (auto& nm : m.names) x.push_back(by_name.at(nm));
		scores[mi][i] = m.predict(x);
		if (features) { // only what was computed
			for (auto& kv : by_name) {
				const std::string& k = kv.first;
				const bool on = k.rfind("VMAF_integer_feature_vif", 0) == 0 ? m.vif
					: k == "VMAF_integer_feature_adm3_score" || k == "VMAF_integer_feature_aim_score" ? m.aim
					: k.rfind("VMAF_integer_feature_adm", 0) == 0 ? m.adm
					: k.rfind("VMAF_integer_feature_motion", 0) == 0 ? m.motion
					: k.rfind("Cambi", 0) == 0 ? m.cambi : m.speed;
				if (on) (*features)[mi][i][k] = kv.second;
			}
			if (m.adm) for (int s = 0; s < 4; s++) { // as libvmaf's ADM writes them with debug=true
				(*features)[mi][i]["VMAF_integer_feature_adm_num_scale" + std::to_string(s) + "_score"] = f.adm_num[s];
				(*features)[mi][i]["VMAF_integer_feature_adm_den_scale" + std::to_string(s) + "_score"] = f.adm_den[s];
			}
		}
	}
	}
	return true;
}

#else // VES_NO_VULKAN

GpuVmaf::GpuVmaf() : impl(nullptr) {}
GpuVmaf::~GpuVmaf() {}
bool GpuVmaf::supported(const std::vector<std::string>&, int, const GpuVmafFrame&, std::string& why) { why = "built without Vulkan"; return false; }
std::unique_ptr<GpuVmaf> GpuVmaf::create(const std::vector<std::string>&, int, const GpuVmafFrame&, std::string& why) { why = "built without Vulkan"; return nullptr; }
bool GpuVmaf::add(const uint8_t* const[3], const size_t[3], const uint8_t* const[3], const size_t[3]) { return false; }
bool GpuVmaf::finish(std::vector<std::vector<double>>&, std::vector<std::vector<std::map<std::string, double>>>*) { return false; }

#endif
