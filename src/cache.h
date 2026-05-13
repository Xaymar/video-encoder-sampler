#pragma once
// The result of a run is a file of its own (options, bitrates, GPU and the VMAF score of every frame), kept next to its encoded stream in the output
// folder; when a configuration comes up again for the same input, that file is what says it was already evaluated. Without an output folder the
// files go to the cache folder instead, so that nothing is run twice then either.
#include <string>
#include <vector>

#include "encoder.h"
#include "convert.h"
#include "vmaf.h"

struct CachedRun {
	double kbps = 0;
	std::vector<double> scores;
	int width = 0, height = 0; // of the encoded video; 0 in files from before it was recorded
	double fps = 0;
	std::string input, video_sha256, reference_sha256; // of the encoded stream and the input; empty if the file predates them; "input" is whatever name was stored
};

// Everything a run's outcome depends on, as one string: the input file (name, size, modification time), the encoder
// and its options, bitrate, key frame interval, geometry, scaler and VMAF settings, and the versions of the libraries
// and the GPU and driver doing the work. Penalties and rankings are applied afterwards and are not part of it.
std::string cache_key(const std::string& input, const std::string& encoder, const std::string& codec, const Options& options,
                      long long bitrate_kbps, double kf, const Shape& shape, const ScaleOptions& scale, const VmafOptions& vmaf);

// The same key in parts (cache_key is these one after the other), for callers that build many keys and want the parts that do not change built once.
std::string key_input(const std::string& input);
std::string key_encoder(const std::string& encoder, const std::string& codec, const Options& options); // with its environment (GPU and driver, library version)
std::string key_run(long long bitrate_kbps, double kf, const Shape& shape);
std::string key_settings(const ScaleOptions& scale, const VmafOptions& vmaf);

// "<label>_<hash>", unique for the run's key: the name of its result file (.json) and stream (.mp4).
std::string result_name(const std::string& label, const std::string& key);
// The hash part alone, for files kept in the cache folder.
std::string key_hash(const std::string& key);

bool result_load(const std::string& file, const std::string& key, CachedRun& out);
// `details` are JSON members ("\"encoder\":\"x264\",...") that go into the file besides the key, the bitrate and the scores.
void result_store(const std::string& file, const std::string& key, const CachedRun& run, const std::string& details);

std::string encoder_library_version(const std::string& encoder); // versions.cpp
std::string decoder_library_version();
std::string scaler_library_version();
std::string vmaf_library_version();
