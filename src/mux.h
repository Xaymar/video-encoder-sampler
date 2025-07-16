#pragma once
// Writes encoded packets into a fragmented MP4 (libavformat): the moov box comes first, then one fragment per key frame.
#include <cstdint>
#include <string>
#include <vector>

#include "picture.h"

// `packets` are in coding order as the encoder produced them (H.264 and HEVC as Annex B, AV1 as temporal units), `pts[i]` is
// the number of the frame in display order, `key[i]` whether the packet starts a key frame. `enc` gives size and frame rate.
bool write_mp4(const std::string& path, Codec codec, const Shape& enc, const Color& color, const std::vector<std::vector<uint8_t>>& packets,
               const std::vector<int64_t>& pts, const std::vector<uint8_t>& key);
