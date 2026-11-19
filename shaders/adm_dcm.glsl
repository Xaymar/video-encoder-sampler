// adm_dcm.comp (one unit, its options in pc) and adm_dcm_multi.comp (ADM_MULTI: every unit of one enhancement gain limit, its csf factors,
// AIM flag and part of tmp in `units`, pc.dbg of them: the bands are read and decoupled once for all)
// Decouple, csf and contrast masking in one pass (adm_decouple.comp, adm_csf.comp, adm_cm.comp, the same integer steps): a workgroup takes a 16x16
// tile of the masking's region, decouples it and its 1 pixel border into shared memory, and writes, per row of the tile, the three inner sums (h, v,
// d) of the DLM and, with pc.aim, of VMAF v1's AIM too, into `tmp` (two ints each). adm_cmsum.comp adds the tiles of each row. Nothing is written
// back into the bands, so every unit can read the same ones.
#include "adm_decouple.glsl"

const int T = 16, E = T + 2;
shared int sX[3][E * E]; // restored
shared int sA[3][E * E]; // additive impairment
shared int64_t sRed[3][T][T];

// The unit's csf factors (pc's, or with ADM_MULTI the table's for the unit being worked on)
int g_rf0, g_rf1, g_rf2;

int csf_dst(int theta, int src) {
	const int rf = theta == 0 ? g_rf0 : theta == 1 ? g_rf1 : g_rf2;
	if (pc.scale == 0) {
		const int shifts = theta == 2 ? 17 : 15;
		const int adds = theta == 2 ? 65535 : 16384;
		return sext16((rf * src + adds) >> shifts);
	}
	return int(((int64_t(uint(rf)) * int64_t(src)) + 134217728l) >> 28);
}
int csf_flt(int dst) {
	if (pc.scale == 0) return sext16((4369 * abs(dst) + 2048) >> 12);
	return int(((143165577l * int64_t(abs(dst))) - 2147483648l) >> 32); // libvmaf's rounding term 1u << 31 in an int32_t: see adm_csf.comp
}
int center(int dst) {
	const int c = abs(dst);
	return pc.scale == 0 ? sext16((8738 * c + 2048) >> 12) : int(((286331153l * int64_t(c)) - 2147483648l) >> 32);
}
int64_t masked(int band, int src, int thr) {
	const int rf = band == 0 ? g_rf0 : band == 1 ? g_rf1 : g_rf2;
	int x;
	if (pc.scale == 0) {
		x = src * rf;
		const int sub = band == 2 ? 12 : 10;
		x = abs(x) - (thr << sub);
		x = x < 0 ? 0 : x;
		const int sq_sh = band == 2 ? 30 : 29;
		const int64_t sq_add = band == 2 ? 536870912l : 268435456l;
		const int x_sq = int(((int64_t(x) * int64_t(x)) + sq_add) >> sq_sh);
		const int cub = band == 0 ? pc.shc0 : band == 1 ? pc.shc1 : pc.shc2;
		return ((int64_t(x_sq) * int64_t(x)) + (1l << (cub - 1))) >> cub;
	}
	x = int(((int64_t(src) * int64_t(uint(rf))) + 134217728l) >> 28);
	x = abs(x) - thr;
	x = x < 0 ? 0 : x;
	const int x_sq = int(((int64_t(x) * int64_t(x)) + 536870912l) >> 30);
	return ((int64_t(x_sq) * int64_t(x)) + (1l << (pc.shc0 - 1))) >> pc.shc0;
}

void main() {
	const int t = int(gl_LocalInvocationIndex);
	const int N = pc.BW * pc.BH;
	const int j0 = pc.scol + int(gl_WorkGroupID.x) * T, i0 = pc.srow + int(gl_WorkGroupID.y) * T;
	for (int e = t; e < E * E; e += T * T) {
		const int i = i0 - 1 + e / E, j = j0 - 1 + e % E;
		int x[3] = int[3](0, 0, 0), a[3] = int[3](0, 0, 0);
		if (i >= pc.dt && i < pc.db && j >= pc.dl && j < pc.dr) { // the region decouple covers (the masking reads only inside it)
			bool angle;
			int kh, r0h;
			decouple(i * pc.BW + j, N, x, a, angle, kh, r0h);
		}
		for (int b = 0; b < 3; b++) { sX[b][e] = x[b]; sA[b][e] = a[b]; }
	}
	barrier();
	const int lx = int(gl_LocalInvocationID.x), ly = int(gl_LocalInvocationID.y);
	const int i = i0 + ly, j = j0 + lx;
	const bool inside = i < pc.erow && j < pc.ecol;
	const int rows = pc.erow - pc.srow, tx = int(gl_NumWorkGroups.x);
#ifdef ADM_MULTI
	for (int u = 0; u < pc.dbg; u++) { // every unit with these decoupled bands (the same enhancement gain limit), one after another
	const ivec4 uu = units[u];
	g_rf0 = uu.x; g_rf1 = uu.y; g_rf2 = uu.z;
	const int unit_aim = uu.w & 1, toff = uu.w >> 1;
#else
	g_rf0 = pc.rf0; g_rf1 = pc.rf1; g_rf2 = pc.rf2;
	const int unit_aim = pc.aim, toff = pc.dbg; // pc.dbg: this unit's part of tmp
#endif
	for (int set = 0; set < (unit_aim != 0 ? 2 : 1); set++) { // 0: DLM (restored, masked by the impairment); 1: AIM (the other way round)
		int64_t val[3] = int64_t[3](0l, 0l, 0l);
		if (inside) {
			int thr = 0;
			for (int theta = 0; theta < 3; theta++) {
				int sum = 0;
				for (int di = 0; di < 3; di++)
					for (int dj = 0; dj < 3; dj++) {
						const int q = (ly + di) * E + lx + dj;
						const int dst = csf_dst(theta, set == 0 ? sA[theta][q] : sX[theta][q]);
						sum += di == 1 && dj == 1 ? center(dst) : csf_flt(dst);
					}
				thr += sum;
			}
			const int q = (ly + 1) * E + lx + 1;
			for (int band = 0; band < 3; band++) val[band] = masked(band, set == 0 ? sX[band][q] : sA[band][q], thr);
		}
		for (int band = 0; band < 3; band++) sRed[band][ly][lx] = val[band];
		barrier();
		if (lx < 3 && i < pc.erow) {
			int64_t sum = 0l;
			for (int k = 0; k < T; k++) sum += sRed[lx][ly][k];
			const int idx = toff + (((set * rows + (i - pc.srow)) * tx + int(gl_WorkGroupID.x)) * 3 + lx) * 2;
			tmp[idx] = int(uint(uint64_t(sum)));
			tmp[idx + 1] = int(uint(uint64_t(sum) >> 32));
		}
		barrier();
	}
#ifdef ADM_MULTI
	}
#endif
}
