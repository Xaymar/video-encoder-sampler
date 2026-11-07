# Result files

Every run VES finishes is stored as one JSON file, `<output>/<input folder>/<label>_<hash>.json`. It is the file the results cache reads back, the file `--upload` sends to a VES website, and
the record a website shows a run from. One file is one run: one encoder with one set of options, on one input, at one bitrate, key frame interval and size.

This describes version 0.2.0. Fields are only added, so a reader should ignore ones it does not know. A field that does not apply to a run is `null`, not left out, unless it says otherwise.

## Fields

| field | type | meaning |
|---|---|---|
| `key` | string | What the run depends on, one `name=value` line each (below). Two runs with the same key are the same run. A site identifies a run by the SHA-256 of this string (UTF-8). |
| `kbps` | number | The bit rate the encoder actually produced, over the whole clip. |
| `input` | string | The input's file name (no folder). |
| `input_url` | string or null | Where the input was downloaded from, when it was a URL. |
| `source_decoder` | string | What decoded the input (`hevc (cuda hardware)`, `h264 (software)`). Empty in a distributed sweep: the host did not decode it. |
| `encoder` | string | `x264`, `x265`, `svtav1`, `nvenc`, `amf` or `qsv`. |
| `codec` | string | `h264`, `hevc` or `av1`. The software encoders make one codec each (x264 `h264`, x265 `hevc`, svtav1 `av1`); a GPU encoder takes it from its config entry. |
| `encoder_version` | string or null | The library behind a **software** encoder, as `<name> <version>`: `x264 build 165  r3222M b35605a`, `x265 4.3+1-e9b8812 [Windows][GCC 16.1.0][64 bit] 8bit+10bit+12bit`, `svt-av1 v4.2.0`. `null` for GPU encoders: they have `gpu_info`. Added after 0.2.0 shipped; older files do not have it. |
| `options` | object | The encoder options of the run, name to string (`{"preset": "p4", "profile": "high"}`). Includes `bitdepth`, `chroma` and `range` when they were set. |
| `width`, `height` | integer | The size that was encoded. |
| `fps` | number | The frame rate that was encoded. |
| `target_kbps` | integer | The bit rate that was asked for. |
| `keyframe_interval_s` | number | The key frame interval that was asked for, in seconds. |
| `gpu` | string or null | `<name> (driver <version>)` and, for a GPU model told apart by PCI id, ` [<vendor>:<device> r<revision>]`. `null` for software encoders. |
| `gpu_info` | object or null | The GPU the run was on: `name`, `driver`, `vendor_id`, `device_id`, `sub_system_id`, `revision` (hex strings). `null` for software encoders. |
| `encoder_caps` | object or null | What the encoder says about itself on that GPU (below). `null` for software encoders and for QuickSync. |
| `video` | string or null | The encoded stream's path under the output folder, if it was kept. |
| `video_sha256` | string or null | SHA-256 (hex) of that stream. |
| `reference_sha256` | string | SHA-256 (hex) of the input file the run was scored against. A website only accepts scores against reference videos it knows. |
| `vmaf.scores` | array of numbers | The VMAF score of every frame, in display order. |
| `vmaf.model` | string | The VMAF model the scores are from, as its file name without `.json` (`vmaf_v1.0.16_1d5h_2160`, `vmaf_4k_v0.6.1neg`). Added after 0.2.0 shipped; older files do not have it, and their `key` has the model on its `vmaf=` line. |
| `vmaf.model_id` | string | The model file's id: a hash of its content (16 hex digits), the one on the key's `vmaf=` line; empty for a model built into libvmaf. Newer files only. |
| `vmaf.chroma_u`, `vmaf.chroma_v` | array of numbers | Only for a model with SpEED chroma (VMAF v1): its U and V values of every frame (lower is better, 0 for no loss). Newer files only. |
| `vmaf.ceiling` | array of numbers | The run's ceiling: what every frame of its feed (input, size, frame rate and pixel format) scores with this model when no encoder is in between. One value per frame, like `scores`. Absent when ves ran with `--no-ceiling`. |
| `vmaf.extra_models` | array of objects | The other models scored in the same pass, when ves was given several (`vmaf.model` as a list): each has `model`, `model_id`, `model_generation`, `scores`, `ceiling` and, for v1, `chroma_u` and `chroma_v`, as above. The key then has a `vmaf_extra=` line naming them. Newer files only. |
| `vmaf.model_generation` | string | `v0` or `v1`, empty if it is not known. The two generations score on different scales, so a site keeps them apart. Older files do not have it; they are v0 unless the `vmaf=` line names a v1 model. |

### Software encoders

`gpu`, `gpu_info` and `encoder_caps` are `null`; what produced the stream is in `encoder_version`. A site that shows "encoder · hardware" for a run can show `encoder_version` where it
would show the GPU. The same string is on the `environment=` line of the key, so older files can be read the same way:

```
environment=svt-av1 v4.2.0
```

### GPU encoders and `encoder_caps`

The same GPU name is not the same encoder: runs are told apart by `gpu_info.vendor_id` and `gpu_info.device_id`. `encoder_caps` tells encoder generations apart, so it is a flat object of
the encoder's own capability names to integers or booleans, and runs whose `encoder_caps` are the same share an encoder generation. It differs per codec, because the encoder reports per codec.

* **NVENC:** the NVENC API's capability values (`NUM_MAX_BFRAMES`, `SUPPORTED_RATECONTROL_MODES`, `SUPPORT_CABAC`, ...), about 60 of them.
* **AMF:** `ACCELERATION_TYPE` first, one of `HARDWARE` (the video block of the GPU), `GPU` (done with GPU compute) or `SOFTWARE`, then the SDK's own capability names and values
  (`MaxBitrate`, `MaxLevel`, `NumOfHwInstances`, `BFrames`, ...). Values that change with the load on the encoder (`RequestedThroughput`) are left out. Older files have `null`.

### `key`

A string of lines, each `name=value`, ending with a newline. A reader that wants one piece takes the first line that starts with the name. The lines:

```
ves-cache-1
input=<file name>|<size>|<time>            a local file; for a download: input=<file name>|url=<url>
encoder=<encoder>/<codec>
environment=<the encoder's library for software encoders; the GPU, driver and PCI id for GPU encoders>
options=<name>=<value>;<name>=<value>;...  sorted by name
bitrate=<kbps> kf=<seconds> shape=<w>x<h>@<fps numerator>/<denominator>
scaling=<up scaler>/<down scaler>  <libplacebo version>
decoder=<ffmpeg version>
vmaf=<libvmaf version> <model> <model hash> [scorer=<device>-1]
vmaf_extra=<model> <model hash>;<model> <model hash>;...   only when more models were scored with it
```

`scorer=` is only there when the GPU scorer produced the scores (`scorer=gpu-1`). The scores of the CPU and GPU scorers agree to about a billionth of a point (before November 2026 the GPU scorer scored v0 models about 4e-4 points low).

## Distributed sweeps

A worker's own result file is not the one that is kept. The host stores each result again under its own key and details, so everything above is as the host knows it: `input` and
`input_url` are the host's, `reference_sha256` is the host's hash of the same bytes, `gpu` and `gpu_info` are the worker's GPU, `encoder_version` and `source_decoder` are the host's
(workers run the same build as the host, and the host did not decode the input). The key and `encoder_version` describe the host's libraries.
