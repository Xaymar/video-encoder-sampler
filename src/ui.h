#pragma once
// Console output while the sweep runs: a live area at the bottom of the terminal (overall progress and one line
// per run in flight, with its stage and how far into the clip it is) and permanent lines above it. When stdout is
// not a console, or --plain is given, it falls back to plain lines.
#include <atomic>
#include <cstdint>
#include <string>

// How far one run has got. Written by the run's threads, read by the UI.
struct RunProgress {
	std::atomic<int> stage{0};            // 0 waiting, 1 encoding, 2 scoring, 3 decoding (the shared decode feeding several runs)
	std::atomic<long long> done{0}, total{0};
	std::atomic<int64_t> since_ms{0};     // when the stage began, steady clock
	// What the stage is doing when it is not working: 0 working, 1 starved (waiting for frames from upstream), 2 backed up (waiting for
	// room downstream). The times add up over the stage; "busy" is time spent in the encoder or scorer itself.
	std::atomic<int> state{0};
	std::atomic<int64_t> busy_ms{0}, starved_ms{0}, blocked_ms{0};

	void begin(int st, long long tot);
	void step() { done++; }
};

namespace ui {
void start(bool plain);
void stop();                                   // clears the live area, restores stderr
bool live();                                   // true when the live area is on

// A run in flight; shown until detach(). Returns a handle.
int attach(RunProgress* p, const std::string& label, const std::string& pool);
void detach(int handle);

void log(const std::string& line);             // a permanent line
void log_plain(const std::string& line);       // only in plain mode: detail the live area already shows
void overall(int done, int total);             // runs finished out of all of them
long long peak_memory_mb();                    // largest working set of this process so far
}
