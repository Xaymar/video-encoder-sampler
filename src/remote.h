#pragma once
// Distributed runs (see docs/distributed.md). The host decides what is run: it cuts the uncached part of a sweep into jobs, each a small config for one input,
// encoder entry, scenario, bitrate, key frame interval and size (and, for GPU encoders, one GPU model), and hands them to workers that connect to it. A worker runs
// its own `ves` on the job and sends the result files back; the host stores them where its own result files go, so the rest of the sweep finds them as finished runs.
#include <functional>
#include <memory>
#include <string>
#include <vector>

using RemoteLog = std::function<void(const std::string&)>;

struct RemoteJobSpec {
	std::string name;    // for messages
	std::string config;  // the sub-config (JSON), without inputs; "@GPU@" stands for the index of the GPU the worker runs it on
	std::string input;   // a path on the host, or an http(s) URL the worker fetches itself
	std::string encoder, codec;
	std::string model;   // GPU encoders: the GPU (as gpu_description() names it) this job must run on; empty: any worker
	struct Run {
		std::string result_file; // where the host keeps this run's result
		std::string key;         // the host's cache key for it, which replaces the worker's in the stored file
	};
	std::vector<Run> runs;
};

struct RemoteHostOptions {
	int port = 0;
	std::string token;        // workers must present it
};

class RemoteHost {
public:
	RemoteHost(const RemoteHostOptions& options, RemoteLog log);
	~RemoteHost();

	bool start(std::string& err);            // listens for workers
	void wait_for_workers(int count);        // blocks until that many are connected

	// A GPU model that connected workers have and that can encode `codec` with `encoder`: its description, its PCI/driver info (JSON) and the encoder capabilities (JSON).
	struct Model {
		std::string description, info, caps;
	};
	std::vector<Model> models(const std::string& encoder, const std::string& codec);

	// Serves `jobs` to the workers until every job is done or given up (three tries each). Returns how many result files were stored.
	size_t run_jobs(const std::vector<RemoteJobSpec>& jobs);

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};

// Connects to the host and runs the jobs it sends, until the process is stopped. `workdir` keeps fetched inputs and the jobs' scratch files; `extra_args` are passed on to
// the `ves` that runs a job (--priority, --hw-decode, ...). The worker tells the host which GPUs it has; `only_gpu` (>= 0) offers just that NVIDIA GPU, so a machine with
// several runs one worker per GPU.
int remote_worker(const std::string& host, int port, const std::string& token, const std::string& workdir, int only_gpu, const std::vector<std::string>& extra_args,
                  const RemoteLog& log);
