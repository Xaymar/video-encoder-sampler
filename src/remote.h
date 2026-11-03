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
	std::string path;    // the input as a file on this machine, when `input` is a URL: the host downloads it once and streams it to the workers
	std::string config;  // the sub-config (JSON), without inputs; "@GPU@" stands for the index of the GPU the worker runs it on
	std::string input;   // a path on the host, or an http(s) URL the worker fetches itself
	std::string encoder, codec;
	// Whether any run of the job can work with a source, given as JSON (what the first job of an input reports); `why` says what is in the way when not.
	// Jobs for which it says no are not sent.
	std::function<bool(const std::string& source_json, std::string& why)> possible;
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
	// `on_stored` is told how many result files have been stored so far, each time one is.
	size_t run_jobs(const std::vector<RemoteJobSpec>& jobs, const std::function<void(size_t)>& on_stored = {});

private:
	struct Impl;
	std::unique_ptr<Impl> impl;
};

// Connects to the host and runs the jobs it sends, until the process is stopped. `workdir` keeps fetched inputs and the jobs' scratch files; `extra_args` are passed on to
// the `ves` that runs a job (--priority, --hw-decode, ...). The worker tells the host which GPUs it has; `only_gpu` (>= 0) offers just that NVIDIA GPU, so a machine with
// several runs one worker per GPU.
// The path of this program.
std::string executable_path();

// What a worker offers: "cpu", or a GPU encoder ("nvenc", "amf", "qsv") with the GPU's index (-1: every GPU of that encoder), and how many encodes run on it at the same
// time (0: not offered, -1: the default).
struct WorkerResource {
	std::string encoder;
	int index = -1;
	int parallel = -1;
};

// `resources` are what is written down; whatever is not in it is detected, and gets `default_parallel` (0: 1 for the CPU, 2 for a GPU).
int remote_worker(const std::string& host, int port, const std::string& token, const std::string& workdir, int only_gpu, int default_parallel, const std::vector<WorkerResource>& resources,
                  const std::vector<std::string>& extra_args, const RemoteLog& log);
