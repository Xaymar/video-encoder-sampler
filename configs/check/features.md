# Feature clip for `ves --check`

A 2 s, 1920x1080, 60 fps clip with a segment for each kind of content that makes encoder features do something. A short piece of real footage on its own may never use some of them, so a "same stream" result could be about the footage rather than the option.

| Time | Content | What it exercises |
|---|---|---|
| 0.0-0.4 s | your footage, unchanged | normal motion |
| 0.4-0.8 s | your footage fading to black | weighted prediction |
| 0.8-1.2 s | hard cut: a still from your footage panning diagonally, next to a slowly changing gradient strip | scene-cut I-frames and lookahead; direct mode, mv_precision and B references (smooth motion); AQ and deblocking (flat area next to detail) |
| 1.2-1.6 s | hard cut: Mandelbrot zoom | zoom motion with fine detail, the 8x8 transform |
| 1.6-2.0 s | hard cut: the footage's first frame, frozen, with a textured box sliding off to the right and back | long-term references and several references (covered background comes back) |

## Making it

`features.ffscript` sits next to this file. Point `-ss` at the part of your footage that you want in the first 0.8 s:

```
ffmpeg -ss 0 -i "Trackmania Nations Forever C04.mp4" -filter_complex_script features.ffscript -map "[out]" -an -c:v libx264 -preset medium -qp 2 -profile:v high -g 120 features.mp4
```

- The output is near-lossless 8-bit H.264 High, so NVDEC can still decode it. True lossless (`-qp 0`) uses the High 4:4:4 Predictive profile, which NVDEC can't decode.
- The Mandelbrot part makes it take a minute or two.
- Use it with `--check=2` so all 2 s get encoded.
