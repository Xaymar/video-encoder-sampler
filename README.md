# Video Encoding Samples
Video Encoding Samples (VES) is a project to determine the ideal (or even best) settings for any available encoder through comparison with [VMAF](https://github.com/Netflix/vmaf). It was previously used to generate the data shown on [the old website](https://ves.xaymar.com/1.0/) which will eventually be replaced by new, up to date information.

VES is a native C++ application. It decodes each input once, encodes it in-process with every combination of options you list, decodes the result again and scores every single frame with VMAF. No `ffmpeg` processes are spawned. For each group of runs that share an input, bitrate and key frame interval it then reports which configurations are the best in at least 95% of the frames.

The Node.js implementation in `source/` is the previous version and no longer maintained.

## Requirements
* Windows 10/11 64 bit
* CMake 3.20 or newer, Ninja, and a C++20 compiler (developed with LLVM clang)
* The third-party libraries listed below
* Any number of input videos (`.mkv`, `.mp4`, `.mov`; 8-bit 4:2:0)
* A lot of time.

## Third-Party Libraries
Only the NVENC and AMF headers are tracked in git. The rest are binaries you place under `third-party/`:

| Directory | Contents | Where it came from |
|---|---|---|
| `third-party/ffmpeg-dev/` | `include/`, `lib/`, `bin/` of a **shared** FFmpeg build with libvmaf | [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) `ffmpeg-release-full-shared` (BtbN builds are not compatible) |
| `third-party/x264/ucrt64/` | `bin/`, `include/`, `lib/` | MSYS2 package `mingw-w64-ucrt-x86_64-libx264` |
| `third-party/x265/ucrt64/` | same, plus `libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll` in `bin/` | MSYS2 packages `mingw-w64-ucrt-x86_64-x265` and the MinGW runtime |
| `third-party/svt-av1/ucrt64/` | same | MSYS2 package `mingw-w64-ucrt-x86_64-svt-av1` |
| `third-party/nv-codec-headers/` | NVENC API headers (tracked) | [FFmpeg/nv-codec-headers](https://github.com/FFmpeg/nv-codec-headers) |
| `third-party/amf-include/` | AMF API headers (tracked) | [GPUOpen AMF](https://github.com/GPUOpen-LibrariesAndSDKs/AMF), `amf/public/include` |

MSYS2 packages are `.pkg.tar.zst` files from <https://repo.msys2.org/mingw/ucrt64/>; Windows' `tar -xf` extracts them. NVENC and AMF are loaded at runtime from the GPU driver, so they need no libraries. x264, x265 and SVT-AV1 are GPL/BSD licensed libraries that end up next to the executable.

## Building
```
cmake -G Ninja -DCMAKE_CXX_COMPILER=clang++ -B build
cmake --build build
```
The DLLs from `third-party` are copied next to `build/ves.exe`.

## Usage
```
ves [config.json]
ves <input> <encoder> <codec> <bitrate-kbps[,..]> [key=value[,value..] ...]
```
Without arguments VES reads `config.json` from the working directory. The second form runs a quick sweep from the command line; `kf=<seconds>`, `pool=`, `parallel=`, `vmaf.model=`, `vmaf.threads=` and `over.penalty=` are understood there, every other key goes to the encoder.

### config.json
```json
{
	"inputs": ["input"],
	"bitrates": [16000, 8000, 6000, 3500],
	"kfintervals": [2.0],
	"over_penalty": 0.025,
	"vmaf": { "model": "version=vmaf_4k_v0.6.1", "threads": 4 },
	"encoders": [
		{
			"enabled": true,
			"encoder": "nvenc",
			"codec": "h264",
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
* `bitrates` (kbps) and `kfintervals` (seconds) are swept as well.
* Every option is a list of values and **every combination is tested**. `null` leaves the option unset for that combination, an empty list ignores the option.
* `pool` / `parallel`: runs in one pool share a device and at most `parallel` of them run at once. Defaults: `nvenc` and `amf` 2, everything else (`cpu`) 1. Different pools run at the same time. Concurrency changes how long the sweep takes, never the scores.

### Encoders
| `encoder` | Backend | Options |
|---|---|---|
| `nvenc` | NVENC API (`codec`: h264, hevc, av1) | `preset` p1-p7, `tune` hq/ll/ull/lossless, `rc` cbr/vbr, `bf`, `rc-lookahead`, `spatial-aq`, `temporal-aq`, `aq-strength`, `b_ref_mode` none/each/middle, `gpu` |
| `amf` | AMF API (`codec`: h264, hevc, av1) | Raw AMF property names, e.g. `QualityPreset`, `BPicturesPattern` (H.264) or `HevcQualityPreset` (HEVC). Values are numeric. `gpu` selects the n-th AMD adapter |
| `x264` | libx264 | `preset`, `tune`, `profile`, plus any x264 long option (`bframes`, `rc-lookahead`, `aq-mode`, ...) |
| `x265` | libx265 | `preset`, `tune`, `profile`, plus any x265 long option |
| `svtav1` | libSvtAv1Enc | any `SvtAv1EncApp` long option (`preset`, `lookahead`, `enable-tf`, ...) |
| anything else | libavcodec encoder of that name (e.g. `libx264`) | Passed to `av_opt_set` |

An unknown option makes that run fail with a message instead of being ignored. There is no QuickSync backend yet.

## How the result is picked
For every group (same input, target bitrate and key frame interval), a run counts a frame when its VMAF is at least as high as every other run's on that frame. Runs that count at least 95% of the frames are marked `*`; if there are none, the closest one is listed on top.

Constant bitrate encoders still miss their target, and spending more bits would win frames unfairly. A run that overshoots its target loses `over_penalty` VMAF points per percent of overshoot (default 0.025, taken from the VMAF gain between 3000 and 6000 kbps on a test clip). Undershooting costs nothing.

## Current Limitations
* 8-bit 4:2:0 input and output only; frames are assumed to arrive at a constant frame rate.
* Resolution scaling and colour overrides from the old `config.json` are not implemented yet.
* VMAF is computed through FFmpeg's `libvmaf` filter and read from its JSON log.
* Input videos should be properly tagged.
