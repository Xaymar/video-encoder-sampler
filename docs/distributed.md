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
3. A worker takes a job, as a small config. It fetches the input from the host if it does not have it (by SHA-256, kept in its `--worker-dir`; URL inputs it downloads itself),
   then runs its own `ves` on the job (a child process, so a crash stays with the job) and sends the result files back.
4. The host stores each result under the name and **key** it would have used for the run itself, so the rest of the sweep finds those runs finished: ranking, best videos,
   uploads and the result files are exactly as in a local sweep. CPU encoder runs no worker produced are run on the host.

A worker that loses its connection mid-job gives the job back; it goes to the next free worker. A job is tried three times.

## GPUs

An encoder generation changes what comes out: the same settings on a 2080 Ti and on a 4080 Super give different streams. So a run is identified by its settings **and the GPU model**:

* For a GPU encoder, the host plans every run **once per GPU model** that connected workers have and that can encode the codec (`nvenc/av1` is only planned for a GPU that can do AV1).
  Two workers with the same model share the work; a worker with another model gets its own copy of the sweep. The ranking lists them as separate entries, each tagged with its GPU.
* The key of such a run holds that GPU's name and driver, not the host's, so cached results are never mixed between models.
* A job goes only to a worker that has its model. The GPU option of the config has no meaning in this mode (the worker's GPU is used) and is dropped.
* A GPU encoder entry with no capable GPU among the workers is skipped, with a line saying so. Runs of a model that no worker ran are reported as ignored and are not run on the host.
* A machine with several GPUs runs one worker per GPU: `ves --worker host:port --worker-gpu 1`. AMF and QuickSync encode on the first GPU of their vendor only.

## How much a worker runs at once

Machines differ in how many encoder sessions they take (an NVENC card may allow 10, another 5 or 3), so each worker sets its own limits:

* `--worker-jobs N`: how many jobs it runs at the same time (default 1). The host keeps it supplied with that many.
* `--parallel N`: how many runs of one job run at the same time (it is passed on to every job, and replaces the host's value). Without it a job uses the host's `parallel`
  for that encoder, or the default of 2 for GPU encoders and 1 for CPU encoders.

The encoders running at once on a worker are about `--worker-jobs` times `--parallel`, and a job has as many runs as its option combinations, so `--parallel` beyond that does nothing.
Several jobs that need the same input fetch it once.

## Inputs

An input travels from the host to a worker the first time a job of that worker needs it, and stays in the worker's `--worker-dir`. The host sends a worker the jobs whose input it already
has before any other, so with several inputs each worker settles on a few of them and fetches each at most once. URL inputs are never sent; every worker downloads them itself.

## Protocol

TCP, each message is a type byte, a 32-bit big-endian length and a payload. `HELLO` (version, token, name, slots, GPUs) in both directions; `JOB` (id, name, config, input) from the host;
`NEED` / `FILE_DATA` / `FILE_END` to fetch an input; `RESULT` (file name and contents) and `DONE` (ok, tail of the log) from the worker. Everything after `JOB` carries the job id, so a worker's jobs share one connection.

## Not done yet

* **No encoded streams** come back (workers run with `--no-videos`), so best-videos mode keeps nothing from remote runs.
* **Pruning is off** with `--host`: it needs the scores of earlier runs, which are on other machines.
* **No encryption.** The token only keeps strangers out; use a VPN or an SSH tunnel on untrusted networks.
* **Inputs are sent whole**, once per worker, from the host; a transfer cannot resume.
* **Workers are expected to be there when the host plans.** A worker that connects later takes jobs it matches but does not add GPU models to the plan.
* **CPU encoders** are keyed by the host's library versions; workers should run the same build.
* **No live view** of the workers' runs on the host; it logs when a job goes out and comes back.
