#define _USE_MATH_DEFINES
#include "vk_vmaf.h"

#ifndef VES_NO_VULKAN
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>

#include "json.h"
#include "vk_compute.h"

// VMAF on Vulkan compute: motion, VIF and ADM on the luma planes (8 bit), and the model's score step on the CPU. Ported from libvmaf's CPU code
// (integer_motion.c, integer_vif.c, integer_adm.c, predict.c); the shaders are in shaders/. The scores match libvmaf's within 1e-4.

namespace {

struct Features {
	double adm2 = 0, adm[4] = {}, vif[4] = {}, sad = 0;
};

namespace vkvmaf_detail {
struct dwt_model_params { float a, k, f0, g[4]; };
static const dwt_model_params threshold0 = {0.495f, 0.466f, 0.401f, {1.501f, 1.0f, 0.534f, 1.0f}};
static const float amplitudes[6][4] = {
	{0.62171, 0.67234, 0.72709, 0.67234}, {0.34537, 0.41317, 0.49428, 0.41317}, {0.18004, 0.22727, 0.28688, 0.22727},
	{0.091401, 0.11792, 0.15214, 0.11792}, {0.045943, 0.059758, 0.077727, 0.059758}, {0.023013, 0.030018, 0.039156, 0.030018}};
inline float dwt_quant_step(const dwt_model_params* params, int lambda, int theta, double adm_norm_view_dist, int adm_ref_display_height) {
	float r = adm_norm_view_dist * adm_ref_display_height * M_PI / 180.0;
	float temp = log10(pow(2.0, lambda + 1) * params->f0 * params->g[theta] / r);
	float Q = 2.0 * params->a * pow(10.0, params->k * temp * temp) / amplitudes[lambda][theta];
	return Q;
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
};
}

class VkVmaf {
public:
	VkVmaf(int W, int H, double adm_egl, double vif_egl) : W(W), H(H), vif_egl(vif_egl) {
		using namespace vkvmaf_detail;
		ysize = (size_t)W * H;
		raw_size = (ysize + 3) / 4 * 4 + 4;
		raw_ref = vk.buffer(raw_size, 0);
		raw_dis = vk.buffer(raw_size, 0);
		// motion
		gx = (W + 15) / 16; gy = (H + 15) / 16;
		mfr[0] = vk.buffer(raw_size, 0); mfr[1] = vk.buffer(raw_size, 0);
		mtmp = vk.buffer(ysize * 4, 1);
		mpart = vk.buffer((size_t)gx * gy * 4, 2);
		for (int i = 0; i < 2; i++) {
			std::vector<Buf*> b{&mfr[i], &mfr[1 - i], &mtmp, &mpart};
			my[i] = vk.pipeline("motion_y", b);
			mx[i] = vk.pipeline("motion_x", b);
		}
		// vif
		for (int s = 0; s < 4; s++) { vw[s] = s ? vw[s - 1] / 2 : W; vh[s] = s ? vh[s - 1] / 2 : H; }
		for (int s = 1; s < 4; s++) { vdr[s] = vk.buffer((size_t)vw[s] * vh[s] * 4 + 16, 1); vdd[s] = vk.buffer((size_t)vw[s] * vh[s] * 4 + 16, 1); }
		for (int s = 0; s < 4; s++) { vgroups[s] = ((vw[s] + 15) / 16) * ((vh[s] + 15) / 16); vpart[s] = vk.buffer((size_t)vgroups[s] * 4 * 8, 2); }
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
			std::vector<Buf*> b{sref(s - 1), sdis(s - 1), &vtmp, &vlog, &vpart[s], &vdr[s], &vdd[s]};
			vsub_v[s] = vk.pipeline("vif_sub_v", b);
			vsub_h[s] = vk.pipeline("vif_sub_h", b);
		}
		for (int s = 0; s < 4; s++) {
			std::vector<Buf*> b{sref(s), sdis(s), &vtmp, &vlog, &vpart[s], &vtmp, &vtmp};
			vstat_v[s] = vk.pipeline("vif_stat_v", b);
			vstat_h[s] = vk.pipeline("vif_stat_h", b);
		}
		// adm
		const double nvd = 3.0;
		const int display_height = 1080;
		noise_weight = 0.03125;
		for (int s = 0; s < 4; s++) {
			sw[s] = s == 0 ? W : bw[s - 1]; sh[s] = s == 0 ? H : bh[s - 1];
			bw[s] = (sw[s] + 1) / 2; bh[s] = (sh[s] + 1) / 2;
		}
		const float cos_1deg_sq = cos(1.0 * M_PI / 180.0) * cos(1.0 * M_PI / 180.0);
		for (int s = 0; s < 4; s++) {
			AdmPC& pc = apc[s];
			memset(&pc, 0, sizeof pc);
			AdmHost& hs = ah[s];
			pc.SW = sw[s]; pc.SH = sh[s]; pc.BW = bw[s]; pc.BH = bh[s]; pc.scale = s;
			pc.prevN = s ? bw[s - 1] * bh[s - 1] : 0;
			pc.egl = adm_egl; pc.cos1sq = cos_1deg_sq;
			const int w = bw[s], h = bh[s];
			int dl = w * 0.1 - 0.5 - 1, dt = h * 0.1 - 0.5 - 1, dr = w - dl + 2, db = h - dt + 2;
			if (dl < 0) dl = 0;
			if (dr > w) dr = w;
			if (dt < 0) dt = 0;
			if (db > h) db = h;
			pc.dl = dl; pc.dt = dt; pc.dr = dr; pc.db = db;
			hs.left = w * 0.1 - 0.5; hs.top = h * 0.1 - 0.5; hs.right = w - hs.left; hs.bottom = h - hs.top;
			if (hs.left <= 0 || hs.top <= 0) throw std::runtime_error("frame too small for the ADM prototype");
			pc.cl = hs.left; pc.ct = hs.top; pc.cr = hs.right; pc.cb = hs.bottom;
			pc.scol = hs.left > 1 ? hs.left : 1; pc.ecol = hs.right < w - 1 ? hs.right : w - 1;
			pc.srow = hs.top > 1 ? hs.top : 1; pc.erow = hs.bottom < h - 1 ? hs.bottom : h - 1;
			const float factor1 = 1.0f / dwt_quant_step(&threshold0, s, 1, nvd, display_height);
			const float factor2 = 1.0f / dwt_quant_step(&threshold0, s, 2, nvd, display_height);
			hs.rfactor[0] = factor1; hs.rfactor[1] = factor1; hs.rfactor[2] = factor2;
			if (s == 0) {
				hs.i_rfactor[0] = 36453; hs.i_rfactor[1] = 36453; hs.i_rfactor[2] = 49417;
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
		for (int s = 0; s < 4; s++) {
			const size_t N = (size_t)bw[s] * bh[s];
			acur[s] = vk.buffer(20 * N * 4 + 16, 1);
			acm[s] = vk.buffer((size_t)bh[s] * 3 * 8 + 16, 2);
			aden[s] = vk.buffer((size_t)bh[s] * 3 * 8 + 16, 2);
		}
		atmp = vk.buffer((size_t)4 * bh[0] * W * 4 + 16, 1);
		adiv = vk.buffer(65537 * 4, 0);
		{
			int32_t* d = (int32_t*)adiv.map;
			memset(d, 0, 65537 * 4);
			for (int i = 1; i <= 32768; ++i) { const int32_t recip = (int32_t)(1073741824 / i); d[32768 + i] = recip; d[32768 - i] = 0 - recip; }
		}
		static const char* names[6] = {"adm_dwt_v", "adm_dwt_h", "adm_decouple", "adm_csf", "adm_cm", "adm_den"};
		for (int s = 0; s < 4; s++) {
			std::vector<Buf*> b{&acur[s], &acur[s ? s - 1 : 0], &raw_ref, &raw_dis, &atmp, &adiv, &acm[s], &aden[s]};
			for (int k = 0; k < 6; k++) apipe[s][k] = vk.pipeline(names[k], b);
		}
	}

	// Scores one frame pair (8 bit luma planes, W*H bytes each). `index` 0 has no motion.
	Features process(const uint8_t* ref_y, size_t ref_stride, const uint8_t* dis_y, size_t dis_stride, size_t index) {
		using namespace vkvmaf_detail;
		auto put = [&](void* dst, const uint8_t* src, size_t stride) {
			if (stride == (size_t)W) memcpy(dst, src, ysize);
			else for (int y = 0; y < H; y++) memcpy((uint8_t*)dst + (size_t)y * W, src + (size_t)y * stride, W);
		};
		put(raw_ref.map, ref_y, ref_stride);
		put(raw_dis.map, dis_y, dis_stride);
		const int cur = (int)(index & 1), prev = 1 - cur;
		put(mfr[cur].map, ref_y, ref_stride);
		vk.begin();
		const bool with_motion = index > 0;
		if (with_motion) {
			const int pc[2] = {W, H};
			vk.push(pc, sizeof pc);
			vk.bind(my[prev]); vk.dispatch(gx, gy); vk.barrier();
			vk.bind(mx[prev]); vk.dispatch(gx, gy); vk.barrier();
		}
		auto groups = [](int a, int b) { return std::make_pair((uint32_t)((a + 15) / 16), (uint32_t)((b + 15) / 16)); };
		for (int s = 0; s < 4; s++) {
			if (s > 0) {
				VifPC pc{vw[s - 1], vh[s - 1], s, s == 1 ? 0 : 2, 8, 0, vif_egl};
				vk.push(&pc, sizeof pc);
				vk.bind(vsub_v[s]);
				auto g = groups(vw[s - 1], vh[s - 1] / 2);
				vk.dispatch(g.first, g.second); vk.barrier();
				vk.bind(vsub_h[s]);
				g = groups(vw[s - 1] / 2, vh[s - 1] / 2);
				vk.dispatch(g.first, g.second); vk.barrier();
			}
			VifPC pc{vw[s], vh[s], s, s == 0 ? 0 : 2, 8, 0, vif_egl};
			vk.push(&pc, sizeof pc);
			auto g = groups(vw[s], vh[s]);
			vk.bind(vstat_v[s]); vk.dispatch(g.first, g.second); vk.barrier();
			vk.bind(vstat_h[s]); vk.dispatch(g.first, g.second); vk.barrier();
		}
		for (int s = 0; s < 4; s++) {
			AdmPC pc = apc[s];
			vk.push(&pc, sizeof pc);
			vk.bind(apipe[s][0]);
			auto g = groups(sw[s], bh[s]);
			vk.dispatch(g.first, g.second); vk.barrier();
			vk.bind(apipe[s][1]);
			g = groups(bw[s], bh[s]);
			vk.dispatch(g.first, g.second); vk.barrier();
			vk.bind(apipe[s][2]); vk.dispatch(g.first, g.second); vk.barrier();
			vk.bind(apipe[s][3]); vk.dispatch(g.first, g.second); vk.barrier();
			vk.bind(apipe[s][4]); vk.dispatch(1, apc[s].erow - apc[s].srow);
			AdmPC pd = apc[s];
			if (s > 0) pd.shc0 = ah[s].den_cub;
			vk.push(&pd, sizeof pd);
			vk.bind(apipe[s][5]); vk.dispatch(1, apc[s].cb - apc[s].ct);
			vk.barrier();
		}
		vk.submit_wait();

		Features f;
		if (with_motion) {
			const uint32_t* p = (const uint32_t*)mpart.map;
			uint64_t sum = 0;
			for (uint32_t k = 0; k < gx * gy; k++) sum += p[k];
			f.sad = (double)sum / 256.0 / ((double)W * H);
		}
		for (int s = 0; s < 4; s++) {
			const int64_t* p = (const int64_t*)vpart[s].map;
			int64_t sum[4] = {0, 0, 0, 0};
			for (uint32_t k = 0; k < vgroups[s]; k++) for (int q = 0; q < 4; q++) sum[q] += p[k * 4 + q];
			const float num = (float)(sum[0] / 2048.0 + (sum[3] - ((sum[2]) / 16384.0) / (65025.0)));
			const float den = (float)(sum[1] / 2048.0 + sum[3]);
			f.vif[s] = (double)(num / den);
		}
		double num = 0, den = 0, num_s[4], den_s[4];
		for (int s = 0; s < 4; s++) {
			const AdmHost& hs = ah[s];
			const int64_t* cm = (const int64_t*)acm[s].map;
			const int64_t* dn = (const int64_t*)aden[s].map;
			int64_t accum[3] = {0, 0, 0};
			const int rows = apc[s].erow - apc[s].srow;
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
			const float num_scale = (parts[0] + parts[1]) + parts[2];
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
			num += num_scale; den += den_scale;
		}
		const double numden_limit = 1e-10 * (W * H) / (1920.0 * 1080.0);
		const double n = num < numden_limit ? 0 : num, d = den < numden_limit ? 0 : den;
		f.adm2 = d == 0.0 ? 1.0 : n / d;
		for (int s = 0; s < 4; s++) f.adm[s] = num_s[s] / den_s[s];
		return f;
	}

private:
	struct AdmHost { int left, top, right, bottom; uint32_t shift_inner; int shift_cub[3]; float rfactor[3]; uint32_t i_rfactor[3]; int den_cub; };
	Vk vk; // first: it releases the buffers and sets below when it goes away
	int W, H;
	double vif_egl, noise_weight = 0.03125;
	size_t ysize = 0, raw_size = 0;
	Buf raw_ref, raw_dis;
	uint32_t gx = 0, gy = 0;
	Buf mfr[2], mtmp, mpart;
	Pipe my[2], mx[2];
	int vw[4], vh[4];
	uint32_t vgroups[4];
	Buf vdr[4], vdd[4], vpart[4], vtmp, vlog;
	Pipe vsub_v[4], vsub_h[4], vstat_v[4], vstat_h[4];
	int sw[4], sh[4], bw[4], bh[4];
	vkvmaf_detail::AdmPC apc[4];
	AdmHost ah[4];
	Buf acur[4], acm[4], aden[4], atmp, adiv;
	Pipe apipe[4][6];
};

// The model file's score step: normalise the features, nu-SVR with an RBF kernel (libsvm), denormalise, clip.
struct Model {
	std::vector<std::string> names;
	std::vector<double> slopes, intercepts;
	std::vector<std::vector<double>> sv;
	std::vector<double> coef;
	double gamma = 0, rho = 0, clip_min = 0, clip_max = 100;
	bool has_p0 = false, has_p1 = false, has_p2 = false, out_lte_in = false, out_gte_in = false, transform = false;
	double p0 = 0, p1 = 0, p2 = 0;

	double predict(const double* feature) const { // feature: raw values in `names` order
		const size_t n = names.size();
		std::vector<double> x(n);
		for (size_t i = 0; i < n; i++) x[i] = slopes[i + 1] * feature[i] + intercepts[i + 1];
		double sum = 0;
		for (size_t k = 0; k < sv.size(); k++) {
			double d2 = 0;
			for (size_t i = 0; i < n; i++) { const double d = x[i] - sv[k][i]; d2 += d * d; }
			sum += coef[k] * exp(-gamma * d2);
		}
		sum -= rho;
		double y = (sum - intercepts[0]) / slopes[0];
		if (transform) { // polynomial mapping and rectification, as in libvmaf's predict.c (no knots)
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
		y = y < clip_min ? clip_min : y;
		y = y > clip_max ? clip_max : y;
		return y;
	}
};

} // namespace

struct GpuVmaf::Impl {
	Model model;
	std::unique_ptr<VkVmaf> gpu;
	std::vector<Features> feats;
};

GpuVmaf::GpuVmaf() : impl(new Impl) {}
GpuVmaf::~GpuVmaf() { delete impl; }

static std::string slurp(const std::string& path) {
	std::ifstream f(path, std::ios::binary);
	if (!f) throw std::runtime_error("cannot read " + path);
	return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

static Model load_model(const std::string& path, double& adm_egl, double& vif_egl) {
	Json j = Json::parse(slurp(path));
	const Json* md = j.find("model_dict");
	if (!md) throw std::runtime_error("not a VMAF model file");
	Model m;
	for (auto& x : md->find("feature_names")->a) m.names.push_back(x.text());
	for (auto& x : md->find("slopes")->a) m.slopes.push_back(atof(x.text().c_str()));
	for (auto& x : md->find("intercepts")->a) m.intercepts.push_back(atof(x.text().c_str()));
	if (auto* c = md->find("score_clip")) { m.clip_min = atof(c->a[0].text().c_str()); m.clip_max = atof(c->a[1].text().c_str()); }
	// libvmaf applies a model's score_transform only when asked to, and ves does not ask, so it is left out here as well.
	if (md->find("norm_type") && md->find("norm_type")->text() != "linear_rescale") throw std::runtime_error("only linear_rescale models are supported");
	adm_egl = vif_egl = 100.0;
	const Json* opts = md->find("feature_opts_dicts");
	for (size_t i = 0; opts && i < opts->a.size(); i++) {
		if (auto* v = opts->a[i].find("adm_enhn_gain_limit")) adm_egl = atof(v->text().c_str());
		if (auto* v = opts->a[i].find("vif_enhn_gain_limit")) vif_egl = atof(v->text().c_str());
	}
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
	static const char* known[] = {"VMAF_integer_feature_adm2_score", "VMAF_integer_feature_motion2_score", "VMAF_integer_feature_vif_scale0_score", "VMAF_integer_feature_vif_scale1_score", "VMAF_integer_feature_vif_scale2_score", "VMAF_integer_feature_vif_scale3_score"};
	for (auto& n : m.names) if (std::find(std::begin(known), std::end(known), n) == std::end(known)) throw std::runtime_error("the model uses a feature the GPU scorer does not have: " + n);
	if (m.slopes.size() != m.names.size() + 1 || m.intercepts.size() != m.names.size() + 1 || m.sv.empty()) throw std::runtime_error("incomplete model");
	return m;
}

bool GpuVmaf::supported(const std::string& model_path, int w, int h, std::string& why) {
	if (!Vk::available(why)) return false;
	if (w < 160 || h < 90) { why = "the frame is too small"; return false; }
	try {
		double a, v;
		load_model(model_path, a, v);
	} catch (const std::exception& e) { why = e.what(); return false; }
	return true;
}

std::unique_ptr<GpuVmaf> GpuVmaf::create(const std::string& model_path, int w, int h, std::string& why) {
	if (!supported(model_path, w, h, why)) return nullptr;
	std::unique_ptr<GpuVmaf> g(new GpuVmaf);
	try {
		double adm_egl, vif_egl;
		g->impl->model = load_model(model_path, adm_egl, vif_egl);
		g->impl->gpu = std::make_unique<VkVmaf>(w, h, adm_egl, vif_egl);
	} catch (const std::exception& e) {
		why = e.what();
		return nullptr;
	}
	return g;
}

bool GpuVmaf::add(const uint8_t* ref, size_t ref_stride, const uint8_t* dis, size_t dis_stride) {
	try {
		impl->feats.push_back(impl->gpu->process(ref, ref_stride, dis, dis_stride, impl->feats.size()));
	} catch (const std::exception& e) {
		fprintf(stderr, "vmaf (GPU): %s\n", e.what());
		return false;
	}
	return true;
}

bool GpuVmaf::finish(std::vector<double>& scores) {
	const size_t n = impl->feats.size();
	scores.assign(n, 0.0);
	for (size_t i = 0; i < n; i++) {
		const Features& f = impl->feats[i];
		// motion2: 0 for the first frame, else the smaller of this frame's sad and the next one's (the last frame: its own)
		const double motion2 = i == 0 ? 0.0 : (i + 1 < n ? std::min(f.sad, impl->feats[i + 1].sad) : f.sad);
		const std::map<std::string, double> by_name = {
			{"VMAF_integer_feature_adm2_score", f.adm2}, {"VMAF_integer_feature_motion2_score", motion2},
			{"VMAF_integer_feature_vif_scale0_score", f.vif[0]}, {"VMAF_integer_feature_vif_scale1_score", f.vif[1]},
			{"VMAF_integer_feature_vif_scale2_score", f.vif[2]}, {"VMAF_integer_feature_vif_scale3_score", f.vif[3]}};
		std::vector<double> x;
		for (auto& nm : impl->model.names) x.push_back(by_name.at(nm));
		scores[i] = impl->model.predict(x.data());
	}
	return true;
}

#else // VES_NO_VULKAN

GpuVmaf::GpuVmaf() : impl(nullptr) {}
GpuVmaf::~GpuVmaf() {}
bool GpuVmaf::supported(const std::string&, int, int, std::string& why) { why = "built without Vulkan"; return false; }
std::unique_ptr<GpuVmaf> GpuVmaf::create(const std::string&, int, int, std::string& why) { why = "built without Vulkan"; return nullptr; }
bool GpuVmaf::add(const uint8_t*, size_t, const uint8_t*, size_t) { return false; }
bool GpuVmaf::finish(std::vector<double>&) { return false; }

#endif
