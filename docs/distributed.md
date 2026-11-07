# Distributed VES

One `ves` is the **host**: it decides what is run. Any number of other machines run `ves --worker` and connect to it, and the host hands them the work.
This is an early version; what it does and what it leaves for later is listed at the end.

```
host:    ves --host 5599 --token secret --config config.jsonc -i videos
worker:  ves --worker host-name:5599 --token secret --hw-decode auto --vmaf-device gpu
```

Workers connect to the host (so they can sit behind a firewall or NAT), may come and go, and need no config file. The host needs an `output` or a cache folder, and
waits for `--min-workers` workers (default 1) before it plans, because the GPUs they have decide what is planned.
Both sides need the same `models/` folder and the same VMAF device (`vmaf.device`, which the host sends along), or scores differ in the last decimals.

## How it works

1. A worker tells the host which GPUs it has when it connects: for NVENC, AMF and QuickSync the GPU's name and driver, and which of H.264, HEVC and AV1 it can encode.
2. The host plans the sweep. Runs whose result file already exists are done; the rest are cut into **jobs**. A job is one input, one encoder entry, one scenario,
   one bitrate, one key frame interval and one size: all the option combinations of that entry, so the runs of a job still share a decode.
3. A worker takes a job, as a small config. It fetches the input from the host if it does not have it (by SHA-256, kept in its `--worker-dir`; a URL input is downloaded by the host once and sent like a file),
   then runs its own `ves` on the job (a child process, so a crash stays with the job) and sends the result files back.
4. The host stores each result under the name and **key** it would have used for the run itself, so the rest of the sweep finds those runs finished: ranking, best videos,
   uploads and the result files are exactly as in a local sweep. CPU encoder runs no worker produced are run on the host.

A job that returns fewer results than it has runs (runs that cannot work with the input, say a frame rate above the source's, are left out without a result) prints the job's last output lines on the host and on the worker, ending with how many runs were ignored and why. Job names carry the bitrate, key frame interval, size and frame rate.

A worker that loses its connection mid-job gives the job back; it goes to the next free worker. A job is tried three times.

## Jobs the input cannot do

The host does not open the inputs, so it does not know what a source allows. The first job of each input goes out alone, and its `ves` tells the host what the source is (size, frame rate, pixel format).
The host then leaves out every queued job of that input whose runs all ask for more than the input has: a higher frame rate, or a size, chroma layout or bit depth above the source's when the scenario
does not say `"upscale": true`. It logs once how many jobs it dropped and why. Other reasons a run cannot work (this GPU lacks a codec, say) are left to the worker, since the host's own GPU says nothing about it.
The runs of dropped jobs count as ignored.

## GPUs

An encoder generation changes what comes out: the same settings on a 2080 Ti and on a 4080 Super give different streams. So a run is identified by its settings **and the GPU model**:

* For a GPU encoder, the host plans every run **once per GPU model** that connected workers have and that can encode the codec (`nvenc/av1` is only planned for a GPU that can do AV1).
  Two workers with the same model share the work; a worker with another model gets its own copy of the sweep. The ranking lists them as separate entries, each tagged with its GPU.
* The key of such a run holds that GPU's name and driver, not the host's, so cached results are never mixed between models.
* A job goes only to a worker that has its model. The GPU option of the config has no meaning in this mode (the worker's GPU is used) and is dropped.
* A GPU encoder entry with no capable GPU among the workers is skipped, with a line saying so. Runs of a model that no worker ran are reported as ignored and are not run on the host.
* A machine with several GPUs runs one worker per GPU: `ves --worker host:port --worker-gpu 1`. AMF is offered for every AMD GPU of the machine (an integrated and a discrete one are two models, each with its own runs); QuickSync encodes on the first Intel GPU only.

## What a worker offers

A worker tells the host what it has and how many encodes may run on each resource at once; the host's config only lists what to test and does not name pools or `parallel`. The easy way is a
config file, `ves --worker worker.jsonc`:

```jsonc
{
  "host": "10.0.0.1:5599", "token": "...", "dir": "ves-worker",
  "resources": {
    "cpu":   { "parallel": 4 },                       // x264, x265, SVT-AV1
    "nvenc": [ { "gpu": 0, "parallel": 10 } ],        // one entry per GPU; without "gpu" it is every GPU of the encoder
    "amf":   { "parallel": 2 }
  }
}
```

* A resource that is left out is detected and gets the default: 1 encode on the CPU, 2 on a GPU. `"parallel": 0` does not offer it. `--worker-jobs N` changes the default.
* Scoring is not counted: it runs on the GPU (`vmaf.device`) and costs little next to an encode. Each worker picks its own scoring GPU: by default the GPU that encodes the job,
  else its first discrete GPU; `--vmaf-gpu <n|name>` on the worker, or `"vmaf_gpu"` in its file, picks another (`ves --vmaf-gpu list` shows the devices). The host's `vmaf.gpu` is not sent.
* A job with several runs (option combinations) takes that many of its resource's encodes, at most all of them, and the worker runs them at the same time. A job waits when its resource has
  too few free, while other resources of the worker keep working, so a worker with an NVIDIA and an AMD GPU uses both.
* `host`, `token`, `dir`, `gpu`, `vmaf_gpu`, `vmaf_device`, `hw_decode`, `decode_threads` and `priority` in the file are the same as `--worker host:port`, `--token`, `--worker-dir`, `--worker-gpu`, `--vmaf-gpu`, `--vmaf-device`, `--hw-decode`, `--decode-threads` and `--priority`; the flags win. `"vmaf": { "device": "gpu", "gpu": "0" }` (as in a sweep config; also `threads`) does the same as `vmaf_device` and `vmaf_gpu`. A worker's device replaces the host's `vmaf.device` for the runs on it. `ves --worker host:port` with no file offers everything detected.
* Several jobs that need the same input fetch it once.
* A worker names its result files and the input's folder by hashes (`--short-names`), so the path stays far below the 260 characters Windows allows however long the input's name or a run's label is. The host matches the results by their options and stores them under its own readable names.
* A job holds only the runs the host still lacks (the rest are in its output folder or cache from earlier sweeps); the worker leaves the others out, so a re-run does not redo finished work.

## Inputs

An input travels from the host to a worker the first time a job of that worker needs it, and stays in the worker's `--worker-dir`. The host sends a worker the jobs whose input it already
has before any other, so with several inputs each worker settles on a few of them and fetches each at most once. A URL input is downloaded by the host once (it needs the file itself) and streamed to the workers like a local file, so every worker has the host's bytes and the server is asked once. A download that is not the size recorded with it is damaged and fetched again.

## Progress

The host's "runs finished" count goes up as results arrive from the workers.

A worker shows the same table as a sweep: its log lines above, and below them the rows of the jobs it runs, with a header that says whether it is connected and how many jobs are running.
The host shows the workers' rows with its own, tagged `[worker name]`; they go away when the job is done or the worker is lost. In plain mode (no terminal) they are listed every 15 seconds like local runs.

What travels is numbers, once a second, per running stage: the stage, the frame count done and in total (bytes for uploads and downloads), the rate, and whether it is working, starved or backed up.
The host and the worker draw the rows from them with the same code. A job's `ves` prints them on a line of its own (`--progress`) and the worker reads that line from the pipe it runs the job on; no file is involved.

## Protocol

TCP, each message is a type byte, a 32-bit big-endian length and a payload. `HELLO` (version, token, name, slots, GPUs) in both directions; `JOB` (id, name, config, input) from the host;
`NEED` / `FILE_DATA` / `FILE_END` to fetch an input; `RESULT` (file name and contents) `PROGRESS` (rows) and `DONE` (ok, tail of the log) from the worker. Everything after `JOB` carries the job id, so a worker's jobs share one connection.

## Not done yet

* **No encoded streams** come back (workers run with `--no-videos`), so best-videos mode keeps nothing from remote runs.
* **Pruning goes in rounds.** The host plans lazily as a sweep on one machine does: the probes of an input go out as jobs first, and each next layer when the workers have sent the scores of the one before (each GPU model of the workers is pruned by its own scores). A round waits for its slowest job, so workers can sit idle at the end of one.
* **No encryption.** The token only keeps strangers out; use a VPN or an SSH tunnel on untrusted networks.
* **Inputs are sent whole**, once per worker, from the host; a transfer cannot resume.
* **Workers are expected to be there when the host plans.** A worker that connects later takes jobs it matches but does not add GPU models to the plan.
* **CPU encoders** are keyed by the host's library versions; workers should run the same build.
