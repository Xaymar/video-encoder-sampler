#pragma once
// Encoders in processes of their own: one for every GPU an encoder runs on ("nvenc" GPU 0, "amf" GPU 1, ...) and one for every CPU encoder library ("x264", "x265",
// "svtav1"). A driver or library that crashes then takes only its own process with it: the runs that were encoding in it fail, the sweep (or the worker) goes on, and the
// next run on that GPU starts a new process. A process that serves one GPU only can also load what belongs to that GPU alone: AMF's runtime from the GPU's own driver
// package (see encoder_amf.cpp), so a GPU whose driver is older than the newest one installed keeps the runtime it was made for.
//
// The encoder in this process is a stand-in that hands the pictures over in shared memory and gets the packets back on a pipe. Decoding, muxing and scoring stay here.
#include <memory>
#include <string>
#include <vector>

#include "encoder.h"

// On by default; off (--in-process) runs every encoder inside this process, as before.
void set_isolated_encoders(bool on);
bool isolated_encoders();
// What the encoder processes are started with besides their own arguments (--priority low, ...).
void set_encoder_process_args(const std::vector<std::string>& args);

// An encoder that runs in the process of its GPU (the "gpu" option at open(), 0 without one) or of its CPU library.
std::unique_ptr<Encoder> make_isolated_encoder(const std::string& name);

// What a GPU encoder can do, asked of the process of that GPU, so that this process never loads the vendor's runtime. Returns false when the question is not one for
// an encoder process (isolation is off, or the encoder runs on the CPU); otherwise `answer` is set, false/"null" when the process could not answer.
bool isolated_supports(const std::string& encoder, const std::string& codec, int chroma, int depth, int gpu, bool& answer);
bool isolated_caps(const std::string& encoder, const std::string& codec, int gpu, std::string& answer);

// Ends the encoder processes (at the end of a sweep); they also end by themselves when this process does.
void stop_encoder_processes();

// `ves --encoder-process <encoder> <gpu> [--priority low]`: the encoder process itself. Talks to the process that started it on stdin and stdout until stdin closes.
int encoder_process_main(int argc, char** argv);
