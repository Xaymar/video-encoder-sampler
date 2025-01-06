#pragma once
// Remembers finished runs on disk so a configuration that was already evaluated for an input is not run again.
#include <string>
#include <vector>

#include "encoder.h"
#include "scaler.h"
#include "vmaf.h"

struct CachedRun {
	double kbps = 0;
	std::vector<double> scores;
};

// Everything a run's outcome depends on, as one string: the input file (name, size, modification time), the encoder
// and its options, bitrate, key frame interval, geometry, scaler and VMAF settings, and the versions of the libraries
// and the GPU and driver doing the work. Penalties and rankings are applied afterwards and are not part of it.
std::string cache_key(const std::string& input, const std::string& encoder, const std::string& codec, const Options& options,
                      long long bitrate_kbps, double kf, const Shape& shape, const ScaleOptions& scale, const VmafOptions& vmaf);

bool cache_load(const std::string& dir, const std::string& key, CachedRun& out);
void cache_store(const std::string& dir, const std::string& key, const CachedRun& run);

std::string encoder_library_version(const std::string& encoder); // versions.cpp
std::string scaler_library_version();
std::string vmaf_library_version();
