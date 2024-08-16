#pragma once
#include <cstdint>
#include <string>
#include <vector>

#include "source.h"

struct VmafOptions {
	std::string model = "version=vmaf_4k_v0.6.1"; // libvmaf model spec: "version=..." or "path=..."
	int threads = 4;
};

// Decodes the encoded packets and scores them against `ref` (which must be a fresh Source of the
// original input) with libvmaf, in-process. `scores` receives one VMAF value per frame.
bool vmaf_score(Source& ref, AVCodecID codec, const std::vector<std::vector<uint8_t>>& packets,
                const VmafOptions& opt, std::vector<double>& scores);
