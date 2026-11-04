# Video Encoding Samples (VES)

VES finds the best settings for a video encoder by measuring them. You give it source videos and lists of encoder options; it encodes every combination, decodes the result again, scores every frame with [VMAF](https://github.com/Netflix/vmaf) and tells you which configurations are best. Supported encoders: NVENC, AMF and QuickSync (H.264, HEVC, AV1), x264, x265 and SVT-AV1.

How it works:

* **One decode, many encodes.** A source is decoded once and fed to all encoders at the same time; each result is scored while it is still being encoded. FFmpeg only reads and decodes video, everything else is native.
* **Per-frame winners.** For each input, bitrate and key frame interval, a configuration counts a frame when its VMAF is at least as high as every other configuration's. Those that count 95% of the frames or more are marked `*`. A run that overshoots its target bitrate loses points on every frame, undershooting is free. The penalty is exponential like the website's: `vmaf.over_penalty` (0.025) points per percent for a hair over, 0.07 at 2% over, 1.9 at 10%, 59 at 20%.
* **Fair inputs.** Scaling and frame rate changes use libplacebo, scoring happens in the source's chroma layout and bit depth, and GPU decoding of sources is only used when it is proven to give exactly the same samples as software. The source's colour range, matrix, primaries and transfer are given to every encoder and end up in the stream, the same for all of them.
* **Everything is kept.** Each run writes its encoded stream (fragmented MP4) and a result file with all per-frame scores; a run that was done before is not repeated.
* **Sharing.** Results can be uploaded to a [VES website](#uploading-results) as they are produced.

## Usage

```
ves                                   # runs config.jsonc
ves --config my.jsonc --input videos  # inputs may be files, folders or http(s) URLs
ves -enc x264 -br 6000 -i clip.mkv --option preset=fast,slow --option max_b_frames=0,2
ves --help
```

* An input given as an http(s) URL is downloaded into the cache folder the first time a run needs it (progress and rate show like the runs do), and not at all when all its results exist already. Results of URL inputs are identified by the URL.
* [`config.jsonc`](config.jsonc) is the commented example to start from; [`config.all-options.jsonc`](config.all-options.jsonc) lists every option of every encoder.
* Every option is a list of values and every combination is tested. `null` leaves an option to the encoder; an unknown name fails the run.
* Option names are snake_case. `max_b_frames`, `lookahead_frames`, `max_ref_frames`, `b_reference_mode` and `rate_control` mean the same on every encoder that has them. `preset`, `profile` (`main`, `high`, ...), `tier` (`main`, `high`) and `level` (`4.1`) use the same names everywhere and are translated to what each encoder's library wants. The rest are each library's own names, converted to snake_case.
* `chroma` (`420`, `444`), `bitdepth` (`8`, `10`) and `range` are lists in a scenario, next to the bitrates; every combination is tested with every encoder. They choose the pixel format the encoder gets; left out, the source's is used. `range` (`source`, `limited`, `full`) is the sample range: the values are converted (full to limited, say) before encoding, the stream is tagged with it, and the decoded result is converted back to the source's range for scoring, so the loss of the conversion counts against that run. RGB sources always become BT.709 YUV, limited unless `range` is `full`. Going above what the source has needs `--upscale` (`"upscale": true` in a scenario). `gpu` picks the GPU of a GPU encoder (`"all"` uses every one).
* Results go to `<output>/<input name>/` (one stream and one `.json` per run) and `<output>/<input name>.json` (the ranking). The cache folder holds downloaded inputs.
* Pruning: an option that shows no effect is not swept further. Within a group, for every combination of the other options, the lowest and the highest value of the option's list (not counting `null`) are run first; when they score the same on every frame, the values in between (and `null`) are skipped and listed under `pruned` in the result file. Because the probes are real runs, an option that only matters in some situations (B-frame settings when B-frames are on) is still found there. `"fixed": ["preset"]` on an encoder entry (`--fixed`, `fixed=`) lists options that are never pruned; `"prune": false` on an entry or in `settings`, `--no-prune` or `prune=off` runs every combination.
* VMAF can be computed on the GPU with Vulkan compute (`"vmaf": {"device": "gpu"}` or `"auto"`, `--vmaf-device`): motion, VIF and ADM are ported from libvmaf's CPU code and the scores match it within 1e-4 VMAF points, at roughly ten times its speed. It needs a Vulkan GPU with 64 bit integers and floats, 8 or 10 bit input and a model file; `auto` falls back to libvmaf otherwise. Scoring follows each run to the GPU that encodes it, and runs of CPU encoders score on the first discrete GPU; `"vmaf": {"gpu": ...}` or `--vmaf-gpu` picks a Vulkan device by number or by part of its name instead (`--vmaf-gpu list` shows them), for machines whose first discrete GPU is not the one to use. The shaders are in `shaders/` (`shaders/build.sh` rebuilds the committed `.spv` files with `glslc`).
* ves runs at idle CPU and disk priority so it does not get in your way; `--priority normal` turns that off.
* Each device works in a process of its own: one per GPU (NVENC GPU 0, AMF GPU 1, ...) and one for the CPU encoders, started when a wave first needs them. A GPU's process does its runs entirely on that GPU: it decodes the input there (with `--hw-decode`, pointed at that very GPU), scales, encodes and scores there, so no frames travel between devices; ves itself only plans and collects the results. A driver or library that crashes takes only its process with it: the runs that were in it fail (each run that shared it with others is run once more on its own, to tell the culprit apart), the next wave starts a new process, and the sweep or the worker goes on. Each AMD GPU's process loads AMF's runtime from that GPU's own driver package, so a GPU whose driver is older than the newest one installed (a Vega iGPU next to a newer dGPU) keeps the runtime that can still encode on it. `--in-process` (`"encoder_processes": false` in `settings`) runs everything in one process, as before.

### Uploading results

Make an API key on the website (*Account → API key*), then set `"upload": {"url": "https://...", "key": "ves_..."}` in the config (or `--upload <url>` and the `VES_API_KEY` environment variable). Each run is uploaded right after it is scored, with its video where the site allows it; runs the site already has are skipped, and a failed upload never fails the sweep.

## Building

Windows: CMake 3.20+, Ninja and clang. The libraries come from MSYS2 packages (staged by [`scripts/stage-msys2.sh`](scripts/stage-msys2.sh)) and a shared FFmpeg development build in `third-party/ffmpeg-dev`; libcurl is a git submodule (`git submodule update --init`). [`.github/workflows/build.yml`](.github/workflows/build.yml) shows every step (job `windows-x64`).

```
cmake -G Ninja -DCMAKE_CXX_COMPILER=clang++ -B build
cmake --build build
```

Linux: a Flatpak, `com.xaymar.ves`, built from [`flatpak/com.xaymar.ves.yml`](flatpak/com.xaymar.ves.yml). GPU encoders on Linux are untested; AMF is Windows only.

Libraries: x264, x265, SVT-AV1, libvmaf, libplacebo (needs a Vulkan driver), FFmpeg, libcurl, and the NVENC, AMF and oneVPL APIs, which are loaded from the GPU driver at run time. VMAF models are in `models/`.

## Limitations

* SVT-AV1 is not deterministic, so its bitrates and scores vary slightly between runs.
* GPUs that return several frames in one packet or one frame in several (AMF before RDNA1, QuickSync before 11th generation, NVENC on Maxwell and older) are not supported.
* 4:1:0, 4:1:1 and 4:4:0 sources are treated as 4:2:0; 16-bit and gray sources are not supported.
* Encoders get one constant frame rate; input videos should be properly tagged.
* Output paths on Windows must stay under 260 characters.
* On Windows, the [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) (x64) has to be installed; it is not part of the archive.

## License

BSD 3-clause, see [LICENSE.md](LICENSE.md).
