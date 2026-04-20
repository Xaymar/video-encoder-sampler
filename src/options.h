#pragma once
// Encoder option names. Every name a user writes is snake_case ("max_b_frames", "enable_ltr"); each backend maps it to the spelling of its own
// library. A few options are common: same name and meaning for every encoder that can do them.
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "encoder.h"

// "enableLTR" -> "enable_ltr", "BRefType" -> "b_ref_type", "aq-mode" -> "aq_mode".
std::string snake_case(const std::string& native);
// "aq_mode" -> "aq-mode", for the libraries whose own names have hyphens (x264, x265, SVT-AV1).
std::string kebab_case(const std::string& snake);

// Option names with the same meaning on every encoder that supports them:
//   max_b_frames        most B-frames in a row (0 = none)
//   lookahead_frames    frames of lookahead for rate control (0 = off)
//   max_ref_frames      most reference frames
//   b_reference_mode    B-frames as references: none | middle | each
//   rate_control        cbr | vbr
// An encoder that cannot do one fails the run with a message.
bool is_common_option(const std::string& key);

// Standard names for the values of "preset", "profile" and "tier" (main | high, for HEVC and AV1), turned into what the encoder's library takes. A profile has the same name for every
// encoder that offers it (h264: baseline, constrained_baseline, main, high, constrained_high, high10, high422, high444; hevc: main, main10, rext;
// av1: main, high, professional); AMF's presets are speed, balanced, quality and high_quality. Anything else is passed on as it is, so the
// encoder's own spellings (numbers, x265's main444_10, ...) still work. Returns false with `error` set for a name the encoder lacks.
bool translate_named_value(const std::string& encoder, const std::string& codec, const std::string& key, std::string& value, std::string& error);

using NativeOptions = std::vector<std::pair<std::string, std::string>>;

// The options of an encoder as its library wants them: common ones translated (the backend's own option names), others through `native_name`.
// Returns false with `error` set for a common option the encoder lacks.
bool resolve_options(const std::string& encoder, const std::string& codec, const Options& options,
                     const std::function<std::string(const std::string&)>& native_name, NativeOptions& out, std::string& error);
