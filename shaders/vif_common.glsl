// VIF (libvmaf integer_vif.c) shared parts: filters, sample access, mirror.
layout(local_size_x = 16, local_size_y = 16) in;

layout(push_constant) uniform PC {
	int W;        // dimensions of the plane this kernel works on (for the subsample kernels: of the source plane)
	int H;
	int scale;    // 0..3: the scale being scored, or being made by a subsample kernel (its filter table)
	int srckind;  // how the source planes are stored: 0 bytes, 1 16 bit, 2 one 32 bit word per sample
	int bpc;      // bits per component of the source video
	int pad;
	double egl;   // vif_enhn_gain_limit
} pc;

layout(std430, binding = 0) readonly buffer RefIn { uint src_ref[]; };
layout(std430, binding = 1) readonly buffer DisIn { uint src_dis[]; };
layout(std430, binding = 2) buffer Tmp { uint tmp[]; };      // arrays of W*H words (the subsample kernels: (H/2)*W)
layout(std430, binding = 3) readonly buffer Log2 { uint log2_table[]; };
layout(std430, binding = 4) buffer Part { int64_t part[]; };  // four sums per workgroup
layout(std430, binding = 5) buffer RefOut { uint out_ref[]; };
layout(std430, binding = 6) buffer DisOut { uint out_dis[]; };

const uint FILT0[17] = uint[17](489u, 935u, 1640u, 2640u, 3896u, 5274u, 6547u, 7455u, 7784u, 7455u, 6547u, 5274u, 3896u, 2640u, 1640u, 935u, 489u);
const uint FILT1[9] = uint[9](1244u, 3663u, 7925u, 12590u, 14692u, 12590u, 7925u, 3663u, 1244u);
const uint FILT2[5] = uint[5](3571u, 16004u, 26386u, 16004u, 3571u);
const uint FILT3[3] = uint[3](10904u, 43728u, 10904u);

int fwidth(int s) { return s == 0 ? 17 : s == 1 ? 9 : s == 2 ? 5 : 3; }
uint coeff(int s, int i) { return s == 0 ? FILT0[i] : s == 1 ? FILT1[i] : s == 2 ? FILT2[i] : FILT3[i]; }

int mirror(int i, int n) {
	if (i < 0) return -i;
	if (i >= n) return 2 * n - i - 2;
	return i;
}

// A sample of a source plane (idx counts samples).
uint px(bool ref, int idx) {
	if (pc.srckind == 0) {
		uint w = ref ? src_ref[idx >> 2] : src_dis[idx >> 2];
		return (w >> ((idx & 3) * 8)) & 255u;
	}
	if (pc.srckind == 1) {
		uint w = ref ? src_ref[idx >> 1] : src_dis[idx >> 1];
		return (w >> ((idx & 1) * 16)) & 65535u;
	}
	return ref ? src_ref[idx] : src_dis[idx];
}
