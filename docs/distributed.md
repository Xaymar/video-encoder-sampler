# Distributed VES

One `ves` is the **host**: it decides what is run. Any number of other machines run `ves --worker` and connect to it, and the host hands them the work.
This is the first slice; what it does and what it leaves for later is listed at the end.

```
host:    ves --host 5599 --token secret --config config.jsonc -i videos
worker:  ves --worker host-name:5599 --token secret --hw-decode auto --vmaf-device gpu
```

Workers connect to the host (so they can sit behind a firewall or NAT), may come and go, and need no config file. The host needs an `output` or a cache folder.
Both sides need the same `models/` folder and the same VMAF device (`vmaf.device`, which the host sends along), or scores differ in the last decimals.

## How it works

1. The host plans the sweep as always. Runs whose result file already exists are done; the rest are cut into **jobs**. A job is one input, one encoder entry, one scenario,
   one bitrate, one key frame interval and one size: all the option combinations of that entry, so the runs of a job still share a decode.
2. A worker takes a job, as a small config. It fetches the input from the host if it does not have it (by SHA-256, kept in its `--worker-dir`; URL inputs it downloads itself),
   then runs its own `ves` on the job (a child process, so a crash stays with the job) and sends the result files back.
3. The host stores each result under the name and **key** it would have used for the run itself, so the rest of the sweep finds those runs finished: ranking, best videos,
   uploads and the result files are exactly as in a local sweep. Anything no worker produced (runs that cannot work with their input, a job given up after three tries) is run on the host.

A worker that loses its connection mid-job gives the job back; it goes to the next free worker. A job is tried three times.

## Protocol

TCP, each message is a type byte, a 32-bit big-endian length and a payload. `HELLO` (version, token, name) in both directions; `JOB` (id, name, config, input) from the host;
`NEED` / `FILE_DATA` / `FILE_END` to fetch an input; `RESULT` (file name and contents) and `DONE` (ok, tail of the log) from the worker.

## Not in this slice

* **No encoded streams** come back (workers run with `--no-videos`), so best-videos mode keeps nothing from remote runs.
* **Pruning is off** with `--host`: it needs the scores of earlier runs, which are on other machines.
* **The key is the host's.** A cache key includes the GPU and driver of the machine that ran it; a remote run is stored under the host's key, so results from workers with different GPUs
  are treated as equal. The result file still names the GPU the worker used.
* **No encryption.** The token only keeps strangers out; use a VPN or an SSH tunnel on untrusted networks.
* **One job per worker at a time**, and jobs are not sized to a worker's GPUs. A worker with several GPUs uses them through the `gpu` option of the job.
* **No live view** of the workers' runs on the host; it logs when a job goes out and comes back.
* **Inputs** are sent whole over the same connection.
