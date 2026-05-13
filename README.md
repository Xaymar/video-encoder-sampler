# Video Encoding Samples (VES)

VES finds the best settings for a video encoder by measuring them. You give it source videos and lists of encoder options; it encodes every combination, decodes the result again, scores every frame with [VMAF](https://github.com/Netflix/vmaf) and tells you which configurations are best. Supported encoders: NVENC, AMF and QuickSync (H.264, HEVC, AV1), x264, x265 and SVT-AV1.

How it works:

* **One decode, many encodes.** A source is decoded once and fed to all encoders at the same time; each result is scored while it is still being encoded. FFmpeg only reads and decodes video, everything else is native.
* **Per-frame winners.** For each input, bitrate and key frame interval, a configuration counts a frame when its VMAF is at least as high as every other configuration's. Those that count 95% of the frames or more are marked `*`. A run that overshoots its target bitrate loses points (`vmaf.over_penalty`), undershooting is free.
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
* ves runs at idle CPU and disk priority so it does not get in your way; `--priority normal` turns that off.

### Uploading results

Make an API key on the website (*Account → API key*), then set `"upload": {"url": "https://...", "key": "ves_..."}` in the config (or `--upload <url>` and the `VES_API_KEY` environment variable). Each run is uploaded right after it is scored, with its video where the site allows it; runs the site already has are skipped, and a failed upload never fails the sweep.

## Building

Windows: CMake 3.20+, Ninja and clang. The libraries come from MSYS2 packages (staged by [`scripts/stage-msys2.sh`](scripts/stage-msys2.sh)) and a shared FFmpeg development build in `third-party/ffmpeg-dev`; libcurl is a git submodule (`git submodule update --init`). [`.github/workflows/windows-x64.yml`](.github/workflows/windows-x64.yml) shows every step.

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

## License

BSD 3-clause, see [LICENSE.md](LICENSE.md).
