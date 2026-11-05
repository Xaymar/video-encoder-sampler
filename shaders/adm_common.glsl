// ADM (libvmaf integer_adm.c), shared parts. Only the "interior" case of the contrast masking is implemented (see adm_test.cpp).
layout(push_constant) uniform PC {
	int SW, SH;          // the plane the DWT reads (scale 0: the frame, else the previous scale's approximation band)
	int BW, BH;          // the bands of this scale: ((SW+1)/2, (SH+1)/2)
	int scale;
	int prevN;           // size of one band block of the previous scale
	double egl;          // adm_enhn_gain_limit
	float cos1sq;        // cos(1 degree)^2, as a float
	int rf0, rf1, rf2;   // csf factors (scale 0: 16 bit values; else 32 bit patterns)
	int dl, dt, dr, db;  // region of decouple and csf
	int cl, ct, cr, cb;  // region of the denominator
	int srow, erow, scol, ecol; // region of the contrast masking
	int shc0, shc1, shc2;       // cubing shifts of the contrast masking (scale 0: per band; else shc0) / of the denominator
	int bpc;                    // bits per sample of the frame (scale 0)
	int aim;                    // csf and contrast masking: 0 for the DLM (the restored part, masked by the additive impairment), 1 for VMAF v1's AIM (the other way round)
} pc;

layout(std430, binding = 0) buffer Cur { int cur[]; };            // 20 blocks of BW*BH
layout(std430, binding = 1) readonly buffer Prev { int prev[]; };
layout(std430, binding = 2) readonly buffer RawRef { uint raw_ref[]; };
layout(std430, binding = 3) readonly buffer RawDis { uint raw_dis[]; };
layout(std430, binding = 4) buffer Tmp { int tmp[]; };            // DWT vertical pass: lo_ref, hi_ref, lo_dis, hi_dis
layout(std430, binding = 5) readonly buffer Div { int div_lookup[]; };
layout(std430, binding = 6) buffer CmRows { int64_t cm_rows[]; };
layout(std430, binding = 7) buffer DenRows { int64_t den_rows[]; };

// blocks of `cur`
const int RA = 0, RH = 1, RV = 2, RD = 3, DA = 4, DH = 5, DV = 6, DD = 7;
const int XH = 8, XV = 9, XD = 10;          // decouple: restored
const int AH = 11, AV = 12, AD = 13;        // decouple: additive impairment
const int CAH = 14, CAV = 15, CAD = 16;     // csf of the additive impairment
const int CFH = 17, CFV = 18, CFD = 19;     // its filtered absolute value

int sext16(int v) { return (v << 16) >> 16; }

// The four source rows (or columns) of output i of a DWT level over n samples.
void src_idx(int i, int n, out int ind[4]) {
	const int hn = (n + 1) / 2;
	if (i == 0) { ind[0] = 1; ind[1] = 0; ind[2] = 1; ind[3] = 2; return; }
	const int b = 2 * i;
	ind[0] = b - 1; ind[1] = b; ind[2] = b + 1; ind[3] = b + 2;
	if (i >= hn - 2) {
		for (int k = 0; k < 4; k++) if (ind[k] >= n) ind[k] = 2 * n - ind[k] - 1;
	}
}

const int LO[4] = int[4](15826, 27411, 7345, -4240);
const int HI[4] = int[4](-4240, -7345, 27411, -15826);
