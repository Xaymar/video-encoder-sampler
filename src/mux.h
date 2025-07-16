#pragma once
// Writes encoded packets into a fragmented MP4 (libavformat): the moov box comes first, then one fragment per key frame.
#include <cstdint>
#include <string>
#include <vector>

#include "picture.h"

// The packets come from a spool file (spool.h), in coding order as the encoder produced them (H.264 and HEVC as Annex B, AV1 as
// temporal units) with their frame number in display order and whether they start a key frame. `enc` gives size and frame rate.
// Only one packet at a time is held in memory.
bool write_mp4(const std::string& path, Codec codec, const Shape& enc, const Color& color, const std::string& spool_path);
