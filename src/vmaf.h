#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "scaler.h"
#include "source.h"

struct VmafOptions {
	std::string model = "version=vmaf_4k_v0.6.1"; // libvmaf model spec: "version=..." or "path=..."
	int threads = 4;
};

// Decodes the encoded packets, scales them back to the original resolution, holds frames up to the original
// frame rate and scores them against `ref` (a fresh Source of the original input) with libvmaf, in-process.
// `enc` is the geometry the packets were encoded at. `scores` receives one VMAF value per original frame.
bool vmaf_score(Source& ref, AVCodecID codec, const std::vector<std::vector<uint8_t>>& packets, const Shape& enc,
                const VmafOptions& opt, const ScaleOptions& scale, std::vector<double>& scores);
