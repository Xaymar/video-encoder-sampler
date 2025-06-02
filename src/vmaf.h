#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "convert.h"
#include "ui.h"
#include "picture.h"
#include "source.h"

struct VmafOptions {
	std::string model = "path=models/vmaf_4k_v0.6.1.json"; // "path=<json model>", relative to the working directory, or "version=<builtin>" if libvmaf has it
	int threads = 4;
};

// Decodes the encoded packets, scales them back to the original resolution, holds frames up to the original
// frame rate and scores them against `ref` (a fresh Source of the original input) with libvmaf, in-process.
// `enc` is the geometry and pixel format the packets were encoded at; `shown[k]` is the number of the encoded frame
// that is on screen at original frame k (dropped frames are held). Both sides are compared in the planar YUV
// format `score_fmt`, converting whichever side differs. `scores` receives one VMAF value per original frame.
bool vmaf_score(Source& ref, Codec codec, const std::vector<std::vector<uint8_t>>& packets, const Shape& enc,
                const std::vector<uint32_t>& shown, Fmt score_fmt, const VmafOptions& opt, const ScaleOptions& scale, std::vector<double>& scores, RunProgress* rp = nullptr);
