#pragma once
// Distributed runs (see docs/distributed.md). The host decides what is run: it cuts the uncached part of a sweep into jobs, each a small config for one input,
// encoder entry, scenario, bitrate, key frame interval and size, and hands them to workers that connect to it. A worker runs its own `ves` on the job and sends the
// result files back; the host stores them where its own result files go, so the rest of the sweep finds them as finished runs.
#include <functional>
#include <string>
#include <vector>

struct RemoteJobSpec {
	std::string name;    // for messages
	std::string config;  // the sub-config (JSON), without inputs
	std::string input;   // a path on the host, or an http(s) URL the worker fetches itself
	struct Run {
		std::string result_file; // where the host keeps this run's result
		std::string key;         // the host's cache key for it, which replaces the worker's in the stored file
	};
	std::vector<Run> runs;
};

struct RemoteHostOptions {
	int port = 0;
	std::string token;        // workers must present it
	int min_workers = 1;      // jobs start once this many have connected
};

// Serves `jobs` to the workers that connect on options.port and returns when every job is done or given up (three tries each).
// Returns how many result files were stored.
size_t remote_run_jobs(const std::vector<RemoteJobSpec>& jobs, const RemoteHostOptions& options, const std::function<void(const std::string&)>& log);

// Connects to the host and runs the jobs it sends, until the process is stopped. `workdir` keeps fetched inputs and the jobs' scratch files; `extra_args` are
// passed on to the `ves` that runs a job (--priority, --hw-decode, ...).
int remote_worker(const std::string& host, int port, const std::string& token, const std::string& workdir, const std::vector<std::string>& extra_args,
                  const std::function<void(const std::string&)>& log);
