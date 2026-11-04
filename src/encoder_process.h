#pragma once
// Waves in processes of their own: one for every GPU an encoder runs on ("nvenc" GPU 0, "amf" GPU 1, ...) and one for the CPU encoders (x264, x265, SVT-AV1).
// Each process does all of its part of a wave on its own device: it decodes the input (on that GPU where hardware decoding works), scales and converts it there,
// encodes, and scores (VMAF on that GPU where it is asked for), so no frame crosses from one device to another. This process only plans and collects the results.
// A driver or library that crashes takes only its own process with it: the runs that were in it fail, the sweep (or the worker) goes on, and the next wave on
// that GPU starts a new process. A process that serves one GPU only can also load what belongs to that GPU alone: AMF's runtime from the GPU's own driver
// package (see encoder_amf.cpp), so a GPU whose driver is older than the newest one installed keeps the runtime it was made for.
#include <string>
#include <vector>

#include "pipeline.h"

// On by default; off (--in-process) runs everything inside this process, as before.
void set_isolated_encoders(bool on);
bool isolated_encoders();
// What the processes are started with besides their own arguments (--priority low, --hw-decode auto, --decode-threads n).
void set_encoder_process_args(const std::vector<std::string>& args);

// run_wave() in the processes: the runs of each GPU (and those of the CPU encoders) in the process of that GPU, all parts at the same time. The rows of the live
// display come from the processes; `feed_rp` only gets the time the decodes were held up. `decoders` says what decoded the input in each process.
bool run_wave_isolated(const std::string& input, const SourceInfo& si, std::vector<WaveRun*>& runs, const ScaleOptions& so, const VmafOptions& vo, RunProgress* feed_rp, FeedResult& fr,
                       std::string& decoders);

// What a GPU encoder can do, asked of the process of that GPU, so that this process never loads the vendor's runtime. Returns false when the question is not one for
// an encoder process (isolation is off, or the encoder runs on the CPU); otherwise `answer` is set, false/"null" when the process could not answer.
bool isolated_supports(const std::string& encoder, const std::string& codec, int chroma, int depth, int gpu, bool& answer);
bool isolated_caps(const std::string& encoder, const std::string& codec, int gpu, std::string& answer);

// Ends the processes (at the end of a sweep); they also end by themselves when this process does.
void stop_encoder_processes();

// `ves --encoder-process <encoder|cpu> <gpu> [--priority low] [--hw-decode <type>] [--decode-threads <n>]`: the process itself. Talks to the process that started it
// on stdin and stdout until stdin closes.
int encoder_process_main(int argc, char** argv);
