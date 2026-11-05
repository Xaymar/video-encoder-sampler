// CAMBI (libvmaf cambi.c), shared parts: the banding feature of VMAF v1, on the distorted frame only. Images are one uint per sample.
layout(push_constant) uniform PC {
	int W, H;          // the source plane (cambi_pre: the scored frame) or the image of this pass
	int OW, OH;        // the output (cambi_pre: the encoded size; cambi_decimate: the next scale)
	int bpc;           // cambi_pre: bits per sample of the scored frame
	int anti_dither;   // cambi_pre/cambi_dither: the encode had fewer than 10 bits
	int mask_index;    // cambi_mask: more zero derivatives than this in the 7x7 window make a pixel "flat"
	int pad;           // cambi_cvalue: half the window
	int nd;            // number of contrast steps (1 << max_log_contrast)
	int vlt_luma;      // luma below which banding is not visible
	int band_base, band_size; // the only values that can contribute
} pc;

int mirror_idx(int i, int n) { return i < 0 ? -i : (i >= n ? 2 * n - i - 2 : i); }
