# Video Encoding Samples
Video Encoding Samples (VES) is a project to determine the ideal (or even best) settings for any available encoder through comparison with [VMAF](https://github.com/Netflix/vmaf). It was previously used to generate the data shown on [the old website](https://ves.xaymar.com/1.0/) which will eventually be replaced by new, up to date information.

VES is a native C++ application. It decodes each input once, encodes it in-process with every combination of options you list, decodes the result again and scores every single frame with VMAF. No `ffmpeg` processes are spawned. For each group of runs that share an input, bitrate and key frame interval it then reports which configurations are the best in at least 95% of the frames.

The previous Node.js implementation was removed; it is still available in the git history before the "Full rebuild" commit.

## Requirements
* Windows 10/11 64 bit
* CMake 3.20 or newer, Ninja, and a C++20 compiler (developed with LLVM clang)
* The third-party libraries listed below
* A GPU with Vulkan support when scaling or frame rate conversion is used (libplacebo)
* Any number of input videos (`.mkv`, `.mp4`, `.mov`; YUV 4:2:0, 4:2:2, 4:4:4 or RGB, 8 to 12 bit)
* A lot of time.

## Third-Party Libraries
Only the NVENC and AMF headers are tracked in git. The rest are binaries you place under `third-party/`:

| Directory | Contents | Where it came from |
|---|---|---|
| `third-party/ffmpeg-dev/` | `include/`, `lib/`, `bin/` of a **shared** FFmpeg build (used for reading the inputs and decoding our own output) | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) `ffmpeg-release-full-shared` (BtbN builds are not compatible) |
| `third-party/x264/ucrt64/` | `bin/`, `include/`, `lib/` | MSYS2 package `mingw-w64-ucrt-x86_64-libx264` |
| `third-party/x265/ucrt64/` | same, plus `libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll` in `bin/` | MSYS2 packages `mingw-w64-ucrt-x86_64-x265` and the MinGW runtime |
| `third-party/svt-av1/ucrt64/` | same | MSYS2 package `mingw-w64-ucrt-x86_64-svt-av1` |
| `third-party/vmaf/ucrt64/` | same | MSYS2 package `mingw-w64-ucrt-x86_64-vmaf` (libvmaf; its MinGW runtime DLLs come with the x265 entry) |
| `third-party/libplacebo/ucrt64/` | the same layout, merged from the packages `libplacebo`, `lcms2`, `libdovi`, `shaderc`, `spirv-cross` and `vulkan-headers` | MSYS2 `mingw-w64-ucrt-x86_64-*`; needs a Vulkan driver at run time |
| `third-party/nv-codec-headers/` | NVENC API headers (tracked) | [FFmpeg/nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers) |
| `third-party/amf-include/` | AMF API headers (tracked) | [GPUOpen AMF](https://github.com/GPUOpen-LibrariesAndSDKs/AMF), `amf/public/include` |
| `third-party/libvpl/` | `ucrt64/` of the oneVPL package (QuickSync headers and the `libvpl-2.dll` dispatcher, loaded at runtime) | MSYS2 `mingw-w64-ucrt-x86_64-libvpl` |

MSYS2 packages are `.pkg.tar.zst` files from <https://repo.msys2.org/mingw/ucrt64/>; Windows' `tar -xf` extracts them. NVENC, AMF and QuickSync are loaded at runtime from the GPU driver (QuickSync through `libvpl-2.dll`), so they need no libraries. x264, x265 and SVT-AV1 are GPL/BSD licensed libraries that end up next to the executable.

VMAF models are JSON files in `models/` (from [Netflix/vmaf](https://github.com/Netflix/vmaf/tree/master/model); the MSYS2 libvmaf has no built-in models). `vmaf.model` is `path=<file>`, relative to the working directory.

## Building
```
cmake -G Ninja -DCMAKE_CXX_COMPILER=clang++ -B build
cmake --build build
```
Only the DLLs from `third-party` are copied next to `build/ves.exe` (not the command line tools in those packages). `cmake --install build --prefix <dir>` puts the program, the DLLs, `models/`, `config.jsonc` and this README into one folder.

## Usage
```
ves [options]
ves <input> <encoder> <codec> <bitrate-kbps[,..]> [key=value[,value..] ...]
```
`ves --help` lists everything. The main options:

| Option | |
|---|---|
| `-cfg`, `--config <file>` | config file, `config.jsonc` by default (`config.json` if that is what there is) |
| `-i`, `--input <path>` | an input video or a directory of videos (`.mkv`, `.mp4`, `.mov`); replaces the config file's inputs, can be repeated |
| `-o`, `--output <dir>` | where the results go, `output` by default: the encoded stream of every run in `<dir>/<input name>/` (fragmented `.mp4`, numbered like the runs: the `moov` box comes first and every key frame starts a new fragment, so players and browsers can start right away; the frames carry the encoder's timing, shifted by the B-frame delay) and one result file `<dir>/<input name>.json` per input. `off` writes nothing, `--no-videos` skips the streams |
| `-c`, `--cache <dir>` | where finished runs are kept (`cache`); `off` disables it |
| `--low-memory` | take less memory, at the cost of speed (see below) |
| `--vmaf-model`, `--vmaf-threads`, `--over-penalty` | scoring; `--vmaf-threads` defaults to 0, all cores shared between the runs scored at the same time |
| `-enc`, `-cd`, `-br`, `-res`, `-fps`, `--kf`, `--option key=a,b` ... | a quick run without a config file, e.g. `ves -i clip.mkv -enc x264 -br 3000,6000 --option preset=fast,medium` |

The result file lists, for every input, each group (bitrate target and key frame interval) with all its runs: label, encoder, codec, options, resolution and frame rate, actual bitrate, overshoot penalty, share of frames in which the run is the best, whether it reached the threshold (`winner`), VMAF mean, minimum and 5th percentile, GPU, whether it came from the cache, and the stream file. Runs that failed are listed under `failed`. Runs taken from the cache have no stream, because nothing was encoded. For GPU encoders every run also records `gpu_info` (name, driver and the PCI ids: vendor, device, sub-system and revision) and `encoder_caps` (NVENC: all 60 capability values the SDK defines for the codec, such as the number of B-frames, encoder engines and supported features; `null` for encoders that cannot tell). The generation of the encoder block is not the CUDA generation of the GPU (a newer GPU can carry an older encoder block), so it is not guessed here: a server reading the result files can map the device id and the capability values to the encoder generation.

In a terminal VES shows a live display at the bottom: the number of finished runs, the elapsed time and an estimate of what is left, and one line for every run in flight (including the option screening before the sweep) with its stage (encode or score), a progress bar, the frames done out of the total, the speed, and when it is stalled what it is waiting for: "starved" (waiting for frames from the decoder) or "backed up" (the decoder cannot hand over more because an encoder is slower; the decode lines show this). The shared decode and the reference decode used for scoring have lines of their own. Every run has two lines, one for its encoder and one for its scorer, which follows the encoder's output while it is still being written. The finished line breaks the time of the wave down as well: encoder busy, starved, decoder backed up, scorer starved. A decoder that is mostly backed up means the encoders are the limit; encoders that are mostly starved mean decoding (or converting) is. Finished runs, ignored and failed ones, and everything the libraries print scroll above it. When the output is not a terminal, or with `--plain`, it prints plain lines instead: what is about to happen (inputs, scenarios, encoders, pools, number of runs), a line when a run starts encoding, one when it starts scoring and one when it finishes, and every 15 seconds the progress of the runs in flight. Configurations that cannot work with the input or the encoder (a frame rate above the source's, a pixel format the encoder does not take, a codec the GPU lacks) are reported as ignored, not failed, and listed under `ignored` in the result file. A requested frame rate within half a percent of the source's measured rate counts as the source's own rate.

The short form (and the config file's `"output"` key) work too; the short form writes no files unless `output=<dir>` is given. `kf=<seconds>`, `res=WxH,..`, `fps=..`, `pool=`, `parallel=`, `prune=off`, `fixed=`, `cache=`, `vmaf.model=`, `vmaf.threads=` and `over.penalty=` are understood there.


### config.jsonc

The file may contain comments (`// ...` and `/* ... */`) and trailing commas, which is why it is `.jsonc`; a plain `.json` file works the same. The result files VES writes are plain `.json`, without comments.
```json
{
	"inputs": ["input"],
	"output": "output",
	"kfintervals": [2.0],
	"scenarios": [
		{ "name": "high", "bitrates": [16000, 8000], "resolutions": ["3840x2160", "1920x1080"], "framerates": [60, 30] },
		{ "name": "low", "bitrates": [3500], "resolutions": ["1920x1080", "1280x720"], "framerates": [30] }
	],
	"scaling": { "upscaler": "ewa_lanczos", "downscaler": "ewa_lanczos" },
	"vmaf": { "model": "path=models/vmaf_4k_v0.6.1.json", "threads": 0, "over_penalty": 0.025 },
	"settings": { "cache": "cache", "videos": true, "hw_decode": "auto", "decode_threads": 0, "low_memory": false, "prune": true },
	"encoders": [
		{
			"enabled": true,
			"encoder": "nvenc",
			"codec": "h264",
			"scenarios": ["high", "low"],
			"pool": "nvenc",
			"parallel": 2,
			"options": {
				"preset": ["p7"],
				"bf": [null, 0, 2],
				"rc-lookahead": [32, 0]
			}
		}
	]
}
```
* `inputs`: files or directories (a directory expands to the videos in it).
* A **scenario** is one sub-sweep: its `bitrates` (kbps), `kfintervals` (seconds, defaults to the top level, then 2.0), `resolutions` (`WxH`) and `framerates` (`30`, `29.97` or `30000/1001`; the decimals 23.976, 29.97, 59.94 and 119.88 mean the exact `n*1000/1001` rates) are combined with every combination of every encoder's options. Use several scenarios when some combinations only make sense together, such as 1080p60 at high bitrates and only 720p30 at 3.5 Mbit. Without `scenarios` the top level is the one scenario. Give a scenario a `name` to refer to it from an encoder. Leaving out `resolutions` or `framerates` keeps the source's.
* Every encoder option is a list of values and **every combination is tested**. `null` leaves the option unset for that combination, an empty list ignores the option.
* `scenarios` (per encoder): the names of the scenarios this encoder takes part in, e.g. to keep a slow CPU encoder out of the 4K ones. Leave it out to use all of them; an unknown name is an error.
* `settings.cache`: directory where finished runs are kept, `"cache"` by default; `false` turns it off (see below). `settings.videos` (default `true`) is whether the encoded streams are saved to the `output` folder; the other settings are `hw_decode`, `decode_threads`, `low_memory` and `prune`, described where they apply. The top-level spellings of these keys, from older files, still work.
* `pool` / `parallel`: runs in one pool share a device and at most `parallel` of them run at once. Every GPU gets a pool of its own (`nvenc:0`, `amf:0`, `nvenc:1`, ...) with `parallel` defaulting to 2; everything else (`cpu`) shares one pool with 1. Different pools run at the same time. Concurrency changes how long the sweep takes, never the scores.

### GPUs
`nvenc`, `amf` and `qsv` take a `gpu` option: the number of the GPU among the adapters of that vendor, in Windows adapter order (default `0`). Like any option it can be swept, `"gpu": [0, 1]`, and `"gpu": ["all"]` means every GPU of the vendor. Each GPU is a pool of its own, so several GPUs encode at the same time, and every run lists the GPU it ran on with its name and driver version (NVIDIA's own version number, e.g. `617.42`, with the Windows one in brackets). NVENC finds the CUDA device that is the same GPU by its LUID, so the numbers match Windows' and not CUDA's own order. The cache key includes the GPU and driver, so a driver update recomputes. Scaling and score conversion run on the GPU named in `scaling.gpu`, or libplacebo's choice.

**Pruning.** Many options do nothing on their own, or only together with another one (adaptive B-frames need B-frames and a look-ahead). Before running an encoder entry, VES encodes a few probe runs per input and scenario (at its first bitrate, key frame interval and shape) and compares the output bytes, which are identical when the options had no effect. Each swept option is moved away from its default (the `null` value if the list has one, else its first value), in groups that are halved until the ones that matter are found; options that change nothing are left at their default and not swept. Whatever was left out is tried again with the options found so far switched on, so options that only work together are kept. Presets and tunes set the starting values of everything else, so list them in the entry's `"fixed"` (`fixed=preset,tune` on the command line): fixed options are never left out and each of their combinations is screened separately. `gpu` is always fixed. `"prune": false` on an entry, or in `settings` or `prune=off` turns this off. The encoders must be deterministic for this to work. SVT-AV1 is not (its multi-threaded rate control gives different bytes from run to run, with the stock `SvtAv1EncApp` v4.2.0 too, at least for 1-pass VBR; `lp=1` makes it repeatable), so it is never pruned, and its scores can differ slightly between runs. Limits: an option that only matters at another bitrate or resolution than the first one of the scenario can be left out wrongly, and combinations where an option has no effect (B-frame settings with B-frames off) are still encoded and scored.

### Scaling and frame rates
Encoding above the source resolution (`"resolutions": ["2560x1440"]` for a 720p clip), or in a higher chroma layout or bit depth than the source (`chroma`, `bitdepth` options), is only done in scenarios that say `"upscale": true` (`--upscale` or `upscale=on` on the command line); elsewhere those runs are ignored. Upscaling sometimes gives a better result at the same bitrate. The decoded result is scaled back down to the source resolution for scoring, so the comparison stays fair. Frame rates above the source are not possible, since frames are only dropped, never made up. Resolution changes and format conversions use libplacebo directly (GPU, Vulkan) with the `upscaler` / `downscaler` named in `scaling` (default `ewa_lanczos` for both). `"scaling": {"gpu": "<Vulkan device name>"}` picks the GPU for it by its exact name, e.g. `NVIDIA GeForce RTX 4080 SUPER`; without it libplacebo chooses. Frame rate conversion only drops frames, and it happens before scaling so no time is spent on frames that are thrown away. It follows the real presentation times from the file (every packet's timestamp is read), so variable frame rate sources work: the first frame in each `1/framerate` slot of the output timeline is kept, and bitrates are computed over the real duration. Without `framerates` every frame is kept. Resolutions or frame rates above the source's are rejected.

Colours are never converted: pixel values go through unchanged and the source's range, matrix, primaries and transfer tags are used for scaling, including full range (`pc`, `yuvj420p`) sources. Encoders are not told the range, since it does not influence what they encode.

Scores are always measured against the original: the decoded video is scaled back to the source resolution and each dropped frame is shown again until the next kept frame, so every run has one VMAF value per original frame and runs with different resolutions or frame rates can be compared directly.

### Pixel formats and bit depth
Sources keep their pixel format and sample layout: planar, semi-planar (NV12, P010), packed or planar RGB, 4:2:2, 4:4:4, 8 to 16 bit, with padding bytes or alpha, little or big endian. libplacebo reads whatever the source has, so nothing is repacked on the CPU. Every encoder run can set `chroma` (`420`, `444`) and `bitdepth` (`8`, `10`, `12`) in its options, and like any option they can be swept (`"bitdepth": [8, 10]`). Without them the encoder gets the source's layout and depth, or the closest one it supports (another layout, 4:2:0, then a lower depth). 4:2:2 is not an output format (NVENC only takes it through a proprietary CUDA surface); 4:2:2 sources are encoded as 4:2:0 and scored as 4:4:4. Asking for something the encoder cannot do fails that run. Conversions are done by libplacebo; RGB sources are converted to YUV as BT.709 limited range, 4:4:4.

Quality is measured with both sides in the source's chroma layout at the higher of the source's and the output's bit depth (RGB counts as 4:4:4), so an 8-bit and a 10-bit encode of the same clip are directly comparable.

Formats currently supported by each backend: `x264` 4:2:0 and 4:4:4 at 8 and 10 bit; `nvenc` whatever the GPU reports (asked once per codec): 8-bit 4:2:0 everywhere, 10-bit and 4:4:4 where the codec and GPU allow it, `x265` 4:2:0 and 4:4:4 at 8, 10 and 12 bit; `svtav1` 4:2:0 at 8 and 10 bit (this SVT-AV1 refuses other layouts); `amf` 4:2:0 at 8 bit, and at 10 bit for HEVC (and AV1) where the GPU does it, found by trying once per codec and depth; this Radeon iGPU has no AV1 and no 10-bit H.264.

### Encoders
| `encoder` | Backend | Options |
|---|---|---|
| `nvenc` | NVENC API (`codec`: h264, hevc, av1) | `preset` p1-p7, `tune` hq/ll/ull/lossless, `rc` cbr/vbr, `bf`, `rc-lookahead`, `spatial-aq`, `temporal-aq`, `aq-strength`, `b_ref_mode` none/each/middle, `gpu` |
| `amf` | AMF API (`codec`: h264, hevc, av1) | Raw AMF property names, e.g. `QualityPreset`, `BPicturesPattern` (H.264) or `HevcQualityPreset` (HEVC). Values are numeric. `gpu` selects the n-th AMD adapter |
| `qsv` | oneVPL (`codec`: h264, hevc, av1) | Raw oneVPL field names of `mfxInfoMFX`, `mfxExtCodingOption2` and `mfxExtCodingOption3`, e.g. `TargetUsage`, `GopRefDist`, `LookAheadDepth`, `BRefType`, `MBBRC`. Values are numbers, or `on`/`off`/`adaptive` for the toggles, `best_quality`/`balanced`/`best_speed` for `TargetUsage`, `cbr`/`vbr`/`icq`/... for `RateControlMethod`. `gpu` selects the n-th Intel adapter. **Written against the headers and never run: no Intel GPU was available.** |
| `x264` | libx264 | `preset`, `tune`, `profile`, plus any x264 long option (`bframes`, `rc-lookahead`, `aq-mode`, ...) |
| `x265` | libx265 | `preset`, `tune`, `profile`, plus any x265 long option |
| `svtav1` | libSvtAv1Enc | any `SvtAv1EncApp` long option (`preset`, `lookahead`, `enable-tf`, ...). Runs as CBR with the low-delay prediction structure, which CBR requires; `rc=1 pred-struct=2` gives VBR with the random-access structure |

An unknown option makes that run fail with a message instead of being ignored.

## Result cache
Every finished run is stored in the cache directory (one small file with the per-frame scores and the actual bitrate), and a run whose entry exists is not repeated, marked `(cached)` in the output. Adding a preset, a bitrate or a resolution to a config therefore only computes the new combinations.

An entry is only reused when everything its result depends on is the same: the input file (name, size and modification time), encoder, codec and options, bitrate, key frame interval, resolution and frame rate, scaler and VMAF settings (including the model file's contents), the versions of the encoder library, FFmpeg and libvmaf, and for `nvenc` and `amf` the GPU name and driver version. Delete the directory (or pass `cache=off` on the command line) to recompute everything. The overshoot penalty and the ranking are applied to the stored scores each time, so changing `over_penalty` needs no new runs.

## How the result is picked
For every group (same input, target bitrate and key frame interval), a run counts a frame when its VMAF is at least as high as every other run's on that frame. Runs that count at least 95% of the frames are marked `*`; if there are none, the closest one is listed on top.

Constant bitrate encoders still miss their target, and spending more bits would win frames unfairly. A run that overshoots its target loses `vmaf.over_penalty` VMAF points per percent of overshoot (default 0.025, taken from the VMAF gain between 3000 and 6000 kbps on a test clip). Undershooting costs nothing.

## Hardware decoding of the sources

Software decoding is the slow part with high bitrate sources, and hardware decoding is only worth having if it gives the very same samples, so VES checks that and does not take it on trust. By default (`--hw-decode auto`) the **sources** (the frames the encoders get and the originals the results are scored against) are decoded by the GPU when that is proven to be identical for the file at hand: before a source is decoded that way, its first 120 frames are decoded both ways and compared sample by sample. If they differ, or the GPU cannot decode the stream, the next candidate is tried, and finally software, with a message saying so. `--hw-decode off` (or `"settings": {"hw_decode": false}`) decodes everything in software; a name (`cuda`, `d3d11va`, `d3d12va`, `dxva2`, `qsv`, `amf`, `vaapi`, `vulkan`, or a decoder such as `hevc_cuvid`) tries only that one. The encoded streams that are scored are always decoded in software.

The candidates, in order: FFmpeg's normal decoder running on a hardware device (what `ffmpeg -hwaccels` lists: `cuda`, `d3d11va`, `d3d12va`, `dxva2`, `qsv`, `amf`, `vaapi` for Linux, `vulkan`; not OpenCL, which breaks more often than it works with several GPUs), then the dedicated hardware decoders for the stream's codec (`*_cuvid`, `*_qsv`, `*_amf`: H.264, HEVC, AV1, VP9, VP8, MPEG-1/2/4, VC-1, MJPEG, VVC, depending on the build). The decoder used is shown when a source is opened and recorded as `source_decoder` in the result files.

What the check is for: decoding is exact by specification for the standard codecs, but not every decoder keeps to it. For example, the AMD decoder (`h264_amf`) on the integrated GPU in the development machine decodes 10-bit H.264 and produces different samples than software, and is rejected by the check. The check covers only the first frames of the file, so a defect that appears later would not be seen; a frame count that does not match the file fails the run.

On a 289 Mbps HEVC 4:4:4 1080p120 file with an RTX 4080 SUPER, the encode-and-score wave took 12 s instead of 21 s, with identical scores (13.5 s against 21.4 s including the check).

Semi-planar frames (NV12, P010 and the 4:2:2 and 4:4:4 relatives, which GPUs hand back and some files use) are unpacked to planar on the CPU, exactly, before they go anywhere else. Converting them with libplacebo instead changes sample values slightly, even though only the memory layout differs: the same content stored as NV12 scored 98.095 where the planar original scored 97.733.

## Where FFmpeg is still used
I/O: the MP4 files are memory-mapped: the file is extended in 64 MB steps and packets are copied straight into the mapped window, so the system's memory manager decides what stays in memory and what goes to storage, orders the writes and reads ahead. Muxing runs on its own thread behind a 32 MB queue, so a slow disk does not hold an encoder up. Memory: each encoder's packets are muxed into its fragmented MP4 as they arrive (next to the results, or a scratch file in the temp folder when nothing is saved, deleted afterwards) and each packet is also handed to the run's scorer through a bounded queue, so a run holds no more than a few packets at a time; the decode times come from the encoder (x264, x265 and SVT-AV1 report them; for NVENC, AMF and QuickSync the B-frame depth of the configuration is used, which can put the first frames a few ticks late); each decoder uses every core by default (`--decode-threads <n>` or `settings.decode_threads` sets it; `--low-memory` uses 4), because decoding is what limits high bitrate sources: on a 289 Mbps HEVC 4:4:4 1080p120 file, 4 threads decode at 111 fps and all 16 cores at 247 fps, for about 370 MB more. A 1080p run needs roughly 0.4 to 1 GB, depending on the encoder's own look-ahead, and every extra encoder in a wave and every VMAF thread (about 35 MB each at 1080p) adds to it. `--low-memory` (or `settings.low_memory`) uses 2 VMAF threads and 4 decoder threads: a 5-encoder wave went from 3.8 GB to 3.0 GB, in 18 s instead of 16 s. `--vmaf-threads` and a smaller `parallel` per pool save more. The peak is printed at the end. Decoding is the expensive part in software, so it is shared. Runs that need the same frames (same input, resolution, frame rate and pixel format) are encoded together in a *wave*: one thread decodes, drops frames, scales and converts (FFmpeg and libplacebo) once, and every encoder in the wave takes the frames from a queue of its own. The decoder stays at most 8 frames ahead of the slowest encoder, and the frames are shared between the queues, not copied. A wave holds up to `parallel` runs of every pool (so one NVENC, one AMD and one x264 run at the same time by default); the wave is as slow as its slowest encoder, and the other pools wait for it. Scoring happens at the same time, following each encoder's output as it is written: a second thread decodes the original once for the whole wave and produces its frames in the scoring format, and every run's scorer decodes its own packets and compares them with those frames. So a wave is finished when its slowest encoder and scorer are, there is no separate scoring phase, and the original is decoded twice per wave (once to encode, once to score), which on a high bitrate source is what the wave's time is made of: the 289 Mbps HEVC 4:4:4 1080p120 file took 20.6 s instead of 25 s. Receiving the finished packets has a thread of its own for the asynchronous encoders (NVENC through its completion events, AMF by polling, SVT-AV1 by blocking reads); x264, x265 and QuickSync return their packets from the call that submits.

FFmpeg only reads and decodes video, always with software decoders (hardware decoders are less exact, so they are refused); everything else is native. The encoders, scaling and conversion (libplacebo), scoring (libvmaf) and the sweep logic work on the tool's own types (`src/picture.h`). FFmpeg sits behind two small interfaces in files named `ff_*.cpp`:

| Piece | Interface | What FFmpeg does for it |
|---|---|---|
| `ff_source.cpp` | `Source` (`source.h`) | demuxes and decodes the input videos, whatever container and codec; frames are handed on in the layout the decoder produced |
| `ff_decode.cpp` | `Decoder` (`decode.h`) | decodes our own H.264, HEVC and AV1 output for scoring |

Replacing FFmpeg would mean providing another implementation of those two interfaces (demuxers and decoders for the formats you need, including 10-bit and 4:4:4 H.264/HEVC/AV1 for our own streams); nothing else depends on it. The FFmpeg version is part of the result cache key.

## Current Limitations
* Encoders are told one constant frame rate (the source's, or the measured average for a variable frame rate source) and get frames numbered in order; only the dropping of frames and the scoring follow the real timestamps.
* SVT-AV1 is not deterministic: the same options give slightly different bitrates and scores from run to run (a few tenths of a VMAF point on a short clip). Compare its results with that in mind.
* Old GPUs that hand back several frames in one packet, or one frame in several packets (AMF before RDNA1, QuickSync before 11th generation, NVENC on Maxwell and earlier), are not supported and untested: the MP4 timestamps assume one packet per frame, so they come out wrong there. Fixing that is left to whoever needs those cards.
* 4:1:0, 4:1:1 and 4:4:0 sources are treated as 4:2:0, and 16-bit or gray sources are not supported.
* Input videos should be properly tagged.
