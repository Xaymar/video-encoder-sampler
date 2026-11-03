#include "remote.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <thread>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include "download.h"
#include "encoder.h"
#include "gpu.h"
#include "hash.h"
#include "json.h"
#include "net.h"
#include "ui.h"

namespace fs = std::filesystem;

namespace {

enum : uint8_t { M_HELLO = 1, M_JOB = 2, M_NEED = 3, M_FILE_DATA = 4, M_FILE_END = 5, M_RESULT = 6, M_DONE = 7, M_PROGRESS = 8 };
constexpr int kVersion = 3;
constexpr size_t kChunk = 1 << 20;
constexpr int kTries = 3;

std::string jq(const std::string& s) { // JSON string
	std::string o = "\"";
	for (unsigned char c : s) {
		if (c == '"' || c == '\\') { o += '\\'; o += (char)c; }
		else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
		else o += (char)c;
	}
	return o + "\"";
}

bool flag_set(const std::string& v) { return v == "1" || v == "true"; } // JSON booleans read back as 1 or 0

// "name_1a2b3c4d.json" -> "name": result files of the same run are named alike on both sides except for the hash of the key, which depends on the machine, and the GPU
// index the worker ran on ("_gpu=1"), which the host's own label does not have.
std::string result_prefix(const std::string& file_name) {
	std::string p = file_name;
	if (p.size() > 14 && p.compare(p.size() - 5, 5, ".json") == 0 && p[p.size() - 14] == '_') p.resize(p.size() - 14);
	const size_t g = p.find("gpu=");
	if (g != std::string::npos && g > 0 && (p[g - 1] == '_' || p[g - 1] == ' ')) {
		size_t e = g + 4;
		while (e < p.size() && isdigit((unsigned char)p[e])) e++;
		p.erase(g - 1, e - (g - 1));
	}
	return p;
}

// The stored result starts {"key":"...",...}; the host's own key goes in place of the worker's.
bool replace_key(std::string& content, const std::string& key) {
	static const std::string head = "{\"key\":\"";
	if (content.compare(0, head.size(), head) != 0) return false;
	size_t i = head.size();
	while (i < content.size() && content[i] != '"') i += content[i] == '\\' ? 2 : 1;
	if (i >= content.size()) return false;
	content = "{\"key\":" + jq(key) + content.substr(i + 1);
	return true;
}

std::string read_file(const fs::path& p) {
	std::ifstream f(p, std::ios::binary);
	return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// ---- what a worker has ----

struct Gpu {
	std::string encoder;
	int index = 0;
	std::string description, info;
	std::vector<std::string> codecs;
	std::map<std::string, std::string> caps; // per codec
};

struct Worker {
	std::string name;
	std::vector<Gpu> gpus;
	std::set<std::string> inputs; // inputs it has been sent
	int slots = 1;                // jobs it runs at the same time
};

} // namespace

struct RemoteHost::Impl {
	RemoteHostOptions opt;
	RemoteLog log;
	net::Listener listener;
	std::thread accept_thread;
	std::atomic<bool> stop{false};
	std::mutex m;
	std::condition_variable cv;
	std::vector<std::thread> serve_threads;
	std::vector<std::shared_ptr<Worker>> workers;
	// the jobs being served
	const std::vector<RemoteJobSpec>* jobs = nullptr;
	std::deque<size_t> queue;
	std::vector<int> tries;
	std::map<std::string, std::string> sha_of; // input path -> SHA-256
	size_t left = 0, imported = 0, stored_now = 0;
	std::function<void(size_t)> on_stored;

	std::string input_sha(const std::string& path) {
		{
			std::lock_guard lock(m);
			auto it = sha_of.find(path);
			if (it != sha_of.end()) return it->second;
		}
		const std::string sha = sha256_file(path);
		std::lock_guard lock(m);
		return sha_of[path] = sha;
	}

	// Which of the worker's GPUs runs this job (-1: any worker will do, -2: this worker cannot).
	static int gpu_for(const Worker& w, const RemoteJobSpec& job) {
		if (job.model.empty()) return -1;
		for (auto& g : w.gpus)
			if (g.encoder == job.encoder && g.description == job.model && std::find(g.codecs.begin(), g.codecs.end(), job.codec) != g.codecs.end()) return g.index;
		return -2;
	}

	// The next job for a worker, or false: the queue has none it can run. Prefers a job whose input the worker already has, so inputs travel once per worker.
	bool next_job(const Worker& w, size_t& id, int& gpu) {
		if (!jobs) return false;
		for (int pass = 0; pass < 2; pass++) {
			for (auto it = queue.begin(); it != queue.end(); ++it) {
				const RemoteJobSpec& j = (*jobs)[*it];
				const int g = gpu_for(w, j);
				if (g == -2) continue;
				if (pass == 0 && !w.inputs.count(j.input)) continue;
				id = *it;
				gpu = g;
				queue.erase(it);
				return true;
			}
		}
		return false;
	}

	// Stores a result file the worker sent. Returns whether it belonged to the job.
	bool import_result(const RemoteJobSpec& job, const std::string& name, std::string content) {
		const std::string want = result_prefix(name);
		for (auto& r : job.runs) {
			if (result_prefix(fs::path(r.result_file).filename().string()) != want) continue;
			if (!replace_key(content, r.key)) return false;
			std::error_code ec;
			fs::create_directories(fs::path(r.result_file).parent_path(), ec);
			const fs::path tmp = r.result_file + ".tmp";
			{
				std::ofstream f(tmp, std::ios::binary);
				f << content;
				if (!f) return false;
			}
			fs::rename(tmp, r.result_file, ec);
			return !ec;
		}
		return false;
	}

	// Ids travel as 4 big-endian bytes in front of the binary messages.
	static std::string id_bytes(size_t id) {
		std::string s(4, '\0');
		for (int i = 0; i < 4; i++) s[(size_t)i] = (char)((id >> (24 - 8 * i)) & 255);
		return s;
	}
	static size_t id_of(const std::string& payload) {
		size_t id = 0;
		for (int i = 0; i < 4 && (size_t)i < payload.size(); i++) id = (id << 8) | (uint8_t)payload[(size_t)i];
		return id;
	}

	// One worker's connection: a dispatcher thread keeps it supplied with up to `slots` jobs at a time while this thread reads what comes back.
	void serve(std::unique_ptr<net::Conn> conn_in) {
		std::shared_ptr<net::Conn> conn = std::move(conn_in);
		uint8_t type;
		std::string payload;
		auto worker = std::make_shared<Worker>();
		std::string who = conn->peer;
		try {
			if (!conn->recv_msg(type, payload) || type != M_HELLO) return;
			Json hello = Json::parse(payload);
			if (auto* t = hello.find("token"); (t ? t->text() : std::string()) != opt.token) { log("worker " + who + " refused: wrong token"); return; }
			if (auto* v = hello.find("version"); !v || atoi(v->text().c_str()) != kVersion) { log("worker " + who + " refused: other protocol version"); return; }
			if (auto* n = hello.find("name")) worker->name = n->text();
			if (auto* s = hello.find("slots")) worker->slots = std::max(1, atoi(s->text().c_str()));
			if (auto* gs = hello.find("gpus")) {
				for (auto& g : gs->a) {
					Gpu gpu;
					gpu.encoder = g.find("encoder")->text();
					gpu.index = atoi(g.find("index")->text().c_str());
					gpu.description = g.find("description")->text();
					gpu.info = g.find("info")->text();
					for (auto& c : g.find("codecs")->a) gpu.codecs.push_back(c.text());
					if (auto* caps = g.find("caps")) for (auto& [k, v] : caps->o) gpu.caps[k] = v.text();
					worker->gpus.push_back(std::move(gpu));
				}
			}
			conn->send_msg(M_HELLO, "{}");
		} catch (const std::exception&) { return; }
		who = worker->name + " (" + conn->peer + ")";
		{
			std::lock_guard lock(m);
			workers.push_back(worker);
		}
		cv.notify_all();
		std::string gpus_text;
		for (auto& g : worker->gpus) gpus_text += (gpus_text.empty() ? "" : "; ") + g.encoder + " " + std::to_string(g.index) + ": " + g.description;
		log("worker " + who + " connected, " + std::to_string(worker->slots) + " job" + (worker->slots == 1 ? "" : "s") + " at a time" + (gpus_text.empty() ? "" : ", GPUs: " + gpus_text));

		struct Active { size_t stored = 0; };
		std::map<size_t, Active> active; // jobs this worker has now, guarded by m
		std::mutex send_m;
		std::atomic<bool> dead{false};
		auto send = [&](uint8_t t, const std::string& p) {
			std::lock_guard lock(send_m);
			return conn->send_msg(t, p);
		};
		std::vector<std::thread> senders; // input transfers

		std::thread dispatcher([&] {
			for (;;) {
				size_t id = 0;
				int gpu = -1;
				{
					std::unique_lock lock(m);
					cv.wait(lock, [&] { return stop || dead || ((int)active.size() < worker->slots && next_job(*worker, id, gpu)); });
					if (stop || dead) return;
					tries[id]++;
					active[id] = Active{};
					worker->inputs.insert((*jobs)[id].input); // it is about to get this one
				}
				try {
					const RemoteJobSpec& job = (*jobs)[id];
					std::string sha, size = "0";
					if (!is_url(job.input)) { sha = input_sha(job.input); size = std::to_string(fs::file_size(job.input)); }
					std::string config = job.config;
					for (size_t at; (at = config.find("@GPU@")) != std::string::npos;) config.replace(at, 5, std::to_string(std::max(gpu, 0)));
					const std::string msg = "{\"id\":" + std::to_string(id) + ",\"name\":" + jq(job.name) + ",\"config\":" + jq(config) + ",\"input\":{\"url\":" + (is_url(job.input) ? "true" : "false") +
					                        ",\"name\":" + jq(is_url(job.input) ? job.input : fs::path(job.input).filename().string()) + ",\"sha256\":" + jq(sha) + ",\"size\":" + size + "}}";
					log("job " + job.name + " -> " + who + (job.model.empty() ? "" : " on " + job.model));
					if (!send(M_JOB, msg)) throw std::runtime_error("connection lost");
				} catch (const std::exception&) {
					dead = true; // the reader notices the lost connection and gives the jobs back
					return;
				}
			}
		});

		while (conn->recv_msg(type, payload)) {
			if (type == M_NEED) {
				const size_t id = id_of(payload);
				senders.emplace_back([&, id] {
					const std::string path = (*jobs)[id].input;
					std::ifstream f(path, std::ios::binary);
					std::string chunk(kChunk, '\0');
					while (f) {
						f.read(chunk.data(), (std::streamsize)kChunk);
						const size_t n = (size_t)f.gcount();
						if (n && !send(M_FILE_DATA, id_bytes(id) + chunk.substr(0, n))) return;
					}
					send(M_FILE_END, id_bytes(id));
				});
			} else if (type == M_RESULT) {
				if (payload.size() < 6) continue;
				const size_t id = id_of(payload);
				const size_t nl = (uint8_t)payload[4] | ((uint8_t)payload[5] << 8);
				if (payload.size() < 6 + nl) continue;
				const std::string name = fs::path(payload.substr(6, nl)).filename().string();
				if (id < jobs->size() && import_result((*jobs)[id], name, payload.substr(6 + nl))) {
					size_t now;
					{
						std::lock_guard lock(m);
						active[id].stored++;
						now = ++stored_now;
					}
					if (on_stored) on_stored(now);
				}
			} else if (type == M_PROGRESS) {
				if (payload.size() < 5) continue;
				const size_t id = id_of(payload);
				ui::set_remote(worker->name + "#" + std::to_string(id), "[" + worker->name + "] ", ui::parse_progress(payload.substr(4)));
			} else if (type == M_DONE) {
				bool ok = false;
				std::string log_tail;
				size_t id = 0;
				try {
					Json d = Json::parse(payload);
					id = (size_t)atoll(d.find("id")->text().c_str());
					ok = d.find("ok") && flag_set(d.find("ok")->text());
					if (auto* l = d.find("log")) log_tail = l->text();
				} catch (const std::exception&) { continue; }
				size_t stored = 0;
				{
					std::lock_guard lock(m);
					auto it = active.find(id);
					if (it == active.end()) continue;
					stored = it->second.stored;
					active.erase(it);
					ui::set_remote(worker->name + "#" + std::to_string(id), "", {});
					imported += stored;
					left--;
				}
				const RemoteJobSpec& job = (*jobs)[id];
				log("job " + job.name + " done on " + who + ": " + std::to_string(stored) + " of " + std::to_string(job.runs.size()) + " results" + (ok ? "" : " (the worker reported a failure)"));
				if (!log_tail.empty() && (!ok || stored < job.runs.size())) log(log_tail); // what the job printed last: why runs failed or were left out
				cv.notify_all();
			}
		}

		// The connection is gone: what this worker was running goes back to the queue.
		dead = true;
		cv.notify_all();
		dispatcher.join();
		for (auto& t : senders) t.join();
		std::lock_guard lock(m);
		for (auto& [id, a] : active) {
			ui::set_remote(worker->name + "#" + std::to_string(id), "", {});
			const RemoteJobSpec& job = (*jobs)[id];
			log("job " + job.name + " lost with " + who);
			if (tries[id] < kTries) queue.push_front(id);
			else { left--; log("job " + job.name + " given up after " + std::to_string(kTries) + " tries"); }
		}
		workers.erase(std::remove(workers.begin(), workers.end(), worker), workers.end());
		cv.notify_all();
	}
};

RemoteHost::RemoteHost(const RemoteHostOptions& options, RemoteLog log) : impl(new Impl) {
	impl->opt = options;
	impl->log = std::move(log);
}

RemoteHost::~RemoteHost() {
	impl->stop = true;
	impl->cv.notify_all();
	if (impl->accept_thread.joinable()) impl->accept_thread.join();
	impl->listener.close();
	std::lock_guard lock(impl->m);
	for (auto& t : impl->serve_threads) t.detach(); // a thread that waits on a socket is let go with the process
}

bool RemoteHost::start(std::string& err) {
	if (!impl->listener.open(impl->opt.port, err)) return false;
	impl->accept_thread = std::thread([this] {
		while (!impl->stop) {
			if (auto c = impl->listener.accept(200)) {
				std::lock_guard lock(impl->m);
				impl->serve_threads.emplace_back([this, conn = std::move(c)]() mutable { impl->serve(std::move(conn)); });
			}
		}
	});
	return true;
}

void RemoteHost::wait_for_workers(int count) {
	std::unique_lock lock(impl->m);
	while ((int)impl->workers.size() < count) {
		impl->log("remote: waiting for workers on port " + std::to_string(impl->opt.port) + " (" + std::to_string(impl->workers.size()) + " of " + std::to_string(count) + ")");
		impl->cv.wait_for(lock, std::chrono::seconds(30));
	}
}

std::vector<RemoteHost::Model> RemoteHost::models(const std::string& encoder, const std::string& codec) {
	std::lock_guard lock(impl->m);
	std::vector<Model> out;
	for (auto& w : impl->workers)
		for (auto& g : w->gpus) {
			if (g.encoder != encoder || std::find(g.codecs.begin(), g.codecs.end(), codec) == g.codecs.end()) continue;
			if (std::any_of(out.begin(), out.end(), [&](const Model& m) { return m.description == g.description; })) continue;
			auto c = g.caps.find(codec);
			out.push_back({g.description, g.info, c == g.caps.end() ? "null" : c->second});
		}
	return out;
}

size_t RemoteHost::run_jobs(const std::vector<RemoteJobSpec>& jobs, const std::function<void(size_t)>& on_stored) {
	if (jobs.empty()) return 0;
	{
		std::lock_guard lock(impl->m);
		impl->jobs = &jobs;
		impl->queue.clear();
		for (size_t i = 0; i < jobs.size(); i++) impl->queue.push_back(i);
		impl->tries.assign(jobs.size(), 0);
		impl->left = jobs.size();
		impl->imported = 0;
		impl->stored_now = 0;
		impl->on_stored = on_stored;
	}
	impl->log("remote: " + std::to_string(jobs.size()) + " jobs");
	impl->cv.notify_all();
	std::unique_lock lock(impl->m);
	while (impl->left > 0) {
		if (impl->cv.wait_for(lock, std::chrono::seconds(60)) == std::cv_status::timeout && impl->left > 0)
			impl->log("remote: " + std::to_string(impl->left) + " jobs left, " + std::to_string(impl->workers.size()) + " workers connected");
	}
	impl->jobs = nullptr;
	impl->queue.clear();
	impl->on_stored = nullptr;
	return impl->imported;
}

// ---- worker ----

namespace {

std::string self_exe() {
#ifdef _WIN32
	wchar_t buf[MAX_PATH * 4];
	const DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD)(sizeof buf / sizeof *buf));
	return fs::path(std::wstring(buf, n)).string();
#else
	std::error_code ec;
	return fs::read_symlink("/proc/self/exe", ec).string();
#endif
}

// Runs a program and hands every line it prints (output and errors together) to `on_line`; returns its exit code, or -1 if it could not be started.
int run_process(const std::string& exe, const std::vector<std::string>& args, const std::function<void(const std::string&)>& on_line) {
	std::string pending;
	auto feed = [&](const char* data, size_t n) {
		pending.append(data, n);
		for (size_t at; (at = pending.find('\n')) != std::string::npos;) {
			std::string line = pending.substr(0, at);
			if (!line.empty() && line.back() == '\r') line.pop_back();
			pending.erase(0, at + 1);
			on_line(line);
		}
	};
#ifdef _WIN32
	auto quote = [](const std::string& a) {
		std::string o = "\"";
		for (size_t i = 0; i < a.size(); i++) {
			size_t slashes = 0;
			while (i < a.size() && a[i] == '\\') { slashes++; i++; }
			if (i == a.size()) { o.append(slashes * 2, '\\'); break; }
			if (a[i] == '"') { o.append(slashes * 2 + 1, '\\'); o += '"'; }
			else { o.append(slashes, '\\'); o += a[i]; }
		}
		return o + "\"";
	};
	std::string line = quote(exe);
	for (auto& a : args) line += " " + quote(a);
	const std::wstring cmd = fs::path(line).wstring();
	SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
	HANDLE rd = nullptr, wr = nullptr;
	if (!CreatePipe(&rd, &wr, &sa, 1 << 16)) return -1;
	SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
	STARTUPINFOW si{};
	si.cb = sizeof si;
	si.dwFlags = STARTF_USESTDHANDLES;
	si.hStdOutput = si.hStdError = wr;
	si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
	PROCESS_INFORMATION pi{};
	std::vector<wchar_t> mutable_cmd(cmd.begin(), cmd.end());
	mutable_cmd.push_back(0);
	const BOOL ok = CreateProcessW(nullptr, mutable_cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi);
	CloseHandle(wr);
	if (!ok) { CloseHandle(rd); return -1; }
	char buf[4096];
	DWORD got;
	while (ReadFile(rd, buf, sizeof buf, &got, nullptr) && got) feed(buf, got);
	CloseHandle(rd);
	if (!pending.empty()) on_line(pending);
	WaitForSingleObject(pi.hProcess, INFINITE);
	DWORD code = 1;
	GetExitCodeProcess(pi.hProcess, &code);
	CloseHandle(pi.hProcess);
	CloseHandle(pi.hThread);
	return (int)code;
#else
	int fds[2];
	if (pipe(fds) != 0) return -1;
	const pid_t pid = fork();
	if (pid < 0) { close(fds[0]); close(fds[1]); return -1; }
	if (pid == 0) {
		close(fds[0]);
		dup2(fds[1], 1);
		dup2(fds[1], 2);
		close(fds[1]);
		std::vector<char*> argv;
		argv.push_back(const_cast<char*>(exe.c_str()));
		for (auto& a : args) argv.push_back(const_cast<char*>(a.c_str()));
		argv.push_back(nullptr);
		execv(exe.c_str(), argv.data());
		_exit(127);
	}
	close(fds[1]);
	char buf[4096];
	ssize_t got;
	while ((got = read(fds[0], buf, sizeof buf)) > 0) feed(buf, (size_t)got);
	close(fds[0]);
	if (!pending.empty()) on_line(pending);
	int status = 0;
	waitpid(pid, &status, 0);
	return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

std::string tail_of(const std::string& text, size_t lines) {
	std::vector<std::string> all;
	std::istringstream in(text);
	for (std::string l; std::getline(in, l);) if (!l.empty()) all.push_back(l);
	std::string out;
	for (size_t i = all.size() > lines ? all.size() - lines : 0; i < all.size(); i++) out += all[i] + "\n";
	return out;
}

// The GPUs this machine offers: for each GPU encoder, its GPUs and the codecs each can encode. NVENC is asked per GPU; AMF and QuickSync only encode on the
// first GPU of their vendor, so that is the one offered.
std::string inventory_json(int only_gpu) {
	std::string out = "[";
	bool first = true;
	for (const char* enc : {"nvenc", "amf", "qsv"}) {
		const std::string encoder = enc;
		const unsigned vendor = encoder_vendor(encoder);
		const size_t n = list_gpus(vendor).size();
		for (size_t i = 0; i < n; i++) {
			if (encoder == "nvenc" ? (only_gpu >= 0 && (int)i != only_gpu) : i > 0) continue;
			std::string codecs, caps;
			for (const char* codec : {"h264", "hevc", "av1"}) {
				std::string json = "null";
				bool ok;
				if (encoder == "nvenc") { json = nvenc_caps_json(codec, (int)i); ok = json != "null" && json != "{}"; }
				else if (encoder == "amf") ok = amf_supports(codec, 420, 8);
				else ok = qsv_supports(codec, 420, 8);
				if (!ok) continue;
				codecs += (codecs.empty() ? "" : ",") + jq(codec);
				caps += (caps.empty() ? "" : ",") + jq(codec) + ":" + jq(json);
			}
			if (codecs.empty()) continue;
			out += std::string(first ? "" : ",") + "{\"encoder\":" + jq(encoder) + ",\"index\":" + std::to_string(i) + ",\"description\":" + jq(gpu_description(vendor, (int)i)) +
			       ",\"info\":" + jq(gpu_json(vendor, (int)i)) + ",\"codecs\":[" + codecs + "],\"caps\":{" + caps + "}}";
			first = false;
		}
	}
	return out + "]";
}

// What the reader thread hands to a job: the pieces of an input being fetched.
struct Inbox {
	std::mutex m;
	std::condition_variable cv;
	std::deque<std::pair<uint8_t, std::string>> q;
	bool lost = false;
	void push(uint8_t type, std::string payload) {
		{
			std::lock_guard lock(m);
			q.emplace_back(type, std::move(payload));
		}
		cv.notify_all();
	}
	void close() {
		{
			std::lock_guard lock(m);
			lost = true;
		}
		cv.notify_all();
	}
	bool pop(uint8_t& type, std::string& payload) {
		std::unique_lock lock(m);
		cv.wait(lock, [&] { return !q.empty() || lost; });
		if (q.empty()) return false;
		type = q.front().first;
		payload = std::move(q.front().second);
		q.pop_front();
		return true;
	}
};

std::string id_bytes(size_t id) {
	std::string s(4, '\0');
	for (int i = 0; i < 4; i++) s[(size_t)i] = (char)((id >> (24 - 8 * i)) & 255);
	return s;
}

size_t id_of(const std::string& payload) {
	size_t id = 0;
	for (int i = 0; i < 4 && (size_t)i < payload.size(); i++) id = (id << 8) | (uint8_t)payload[(size_t)i];
	return id;
}

// Fetches an input from the host (if it is not here already) and returns its path.
bool fetch_input(const std::function<bool(uint8_t, const std::string&)>& send, Inbox& inbox, size_t job_id, const fs::path& dir, const std::string& name, const std::string& sha, fs::path& out,
                 std::string& err) {
	std::string safe = fs::path(name).filename().string();
	out = dir / (sha.substr(0, 16) + "_" + safe);
	std::error_code ec;
	if (fs::exists(out, ec)) return true;
	if (!send(M_NEED, id_bytes(job_id) + "{\"sha256\":" + jq(sha) + "}")) { err = "connection lost"; return false; }
	fs::create_directories(dir, ec);
	const fs::path tmp = out.string() + ".part" + std::to_string(job_id);
	std::ofstream f(tmp, std::ios::binary);
	uint8_t type;
	std::string payload;
	for (;;) {
		if (!inbox.pop(type, payload)) { err = "connection lost"; return false; }
		if (type == M_FILE_END) break;
		if (type != M_FILE_DATA) { err = "unexpected message"; return false; }
		f.write(payload.data() + 4, (std::streamsize)payload.size() - 4);
	}
	f.close();
	if (sha256_file(tmp.string()) != sha) { fs::remove(tmp, ec); err = "the input arrived damaged"; return false; }
	fs::rename(tmp, out, ec);
	return !ec;
}

} // namespace

int remote_worker(const std::string& host, int port, const std::string& token, const std::string& workdir, int only_gpu, int slots, const std::vector<std::string>& extra_args,
                  const RemoteLog& log) {
	std::error_code ec;
	fs::create_directories(workdir, ec);
	const std::string exe = self_exe();
	std::string name = "worker";
#ifdef _WIN32
	if (const char* n = getenv("COMPUTERNAME")) name = n;
#else
	if (const char* n = getenv("HOSTNAME")) name = n;
#endif
	if (only_gpu >= 0) name += " GPU " + std::to_string(only_gpu);
	slots = std::max(1, slots);
	const std::string inventory = inventory_json(only_gpu);
	std::mutex fetch_m;
	std::map<std::string, std::shared_ptr<std::mutex>> fetching; // one fetch of an input at a time, however many jobs want it
	for (;;) {
		std::string err;
		auto conn_u = net::connect_to(host, port, err);
		if (!conn_u) {
			log(err + "; trying again in 5 s");
			std::this_thread::sleep_for(std::chrono::seconds(5));
			continue;
		}
		std::shared_ptr<net::Conn> conn = std::move(conn_u);
		if (!conn->send_msg(M_HELLO, "{\"version\":" + std::to_string(kVersion) + ",\"token\":" + jq(token) + ",\"name\":" + jq(name) + ",\"slots\":" + std::to_string(slots) + ",\"gpus\":" + inventory + "}"))
			continue;
		uint8_t type;
		std::string payload;
		if (!conn->recv_msg(type, payload) || type != M_HELLO) {
			log("the host refused this worker (token or version); trying again in 10 s");
			std::this_thread::sleep_for(std::chrono::seconds(10));
			continue;
		}
		log("connected to " + conn->peer + ", " + std::to_string(slots) + " job" + (slots == 1 ? "" : "s") + " at a time");
		std::atomic<int> running_jobs{0};
		auto show_title = [&](bool connected) {
			ui::set_title("ves worker " + name + "  " + (connected ? "connected to " + conn->peer + "  " + std::to_string(running_jobs.load()) + " of " + std::to_string(slots) + " jobs running"
			                                                       : std::string("waiting for the host")));
		};
		show_title(true);

		std::mutex send_m;
		auto send = [&](uint8_t t, const std::string& p) {
			std::lock_guard lock(send_m);
			return conn->send_msg(t, p);
		};
		std::mutex inbox_m;
		std::map<size_t, std::shared_ptr<Inbox>> inboxes;
		std::vector<std::thread> jobs;

		auto run_job = [&](std::string job_payload, std::shared_ptr<Inbox> inbox) {
			Json job = Json::parse(job_payload);
			const size_t id = (size_t)atoll(job.find("id")->text().c_str());
			const std::string jname = job.find("name")->text();
			const Json& in = *job.find("input");
			const fs::path dir = fs::path(workdir) / ("job-" + std::to_string(id));
			const std::string rows_key = "job#" + std::to_string(id);
			running_jobs++;
			show_title(true);
			std::error_code ec2;
			fs::remove_all(dir, ec2);
			fs::create_directories(dir / "out", ec2);
			log("job " + jname);
			std::string input, failure;
			if (flag_set(in.find("url")->text())) {
				// A URL input is downloaded here, once however many jobs want it, into the folder the job's `ves` looks in, so that it finds the file there.
				input = in.find("name")->text();
				std::shared_ptr<std::mutex> one;
				{
					std::lock_guard lock(fetch_m);
					auto& slot = fetching[input];
					if (!slot) slot = std::make_shared<std::mutex>();
					one = slot;
				}
				std::lock_guard lock(*one);
				static std::set<std::string> fetched; // URLs this worker has downloaded in this run; the file is in its cache and is not asked for again
				static std::mutex fetched_m;
				bool have;
				{
					std::lock_guard lock2(fetched_m);
					have = fetched.count(input) > 0;
				}
				if (!have) {
					RunProgress rp;
					rp.begin(5, 0);
					const int handle = ui::attach(&rp, url_file_name(input), "input");
					try {
						fetch_url(input, (fs::path(workdir) / "cache" / "downloads").string(), &rp);
						std::lock_guard lock2(fetched_m);
						fetched.insert(input);
					} catch (const std::exception& e) { failure = e.what(); }
					rp.finish();
					ui::detach(handle);
				}
			} else {
				const std::string sha = in.find("sha256")->text();
				std::shared_ptr<std::mutex> one;
				{
					std::lock_guard lock(fetch_m);
					auto& slot = fetching[sha];
					if (!slot) slot = std::make_shared<std::mutex>();
					one = slot;
				}
				std::lock_guard lock(*one);
				fs::path p;
				if (fetch_input(send, *inbox, id, fs::path(workdir) / "inputs", in.find("name")->text(), sha, p, failure)) input = p.string();
			}
			bool ok = false;
			size_t sent_results = 0;
			std::string log_text = failure;
			if (failure.empty()) {
				{
					std::ofstream f(dir / "config.jsonc", std::ios::binary);
					f << job.find("config")->text();
				}
				std::vector<std::string> args{"--plain", "--progress", "--config", (dir / "config.jsonc").string(), "--input", input, "--output", (dir / "out").string(), "--no-videos",
				                               "--cache", (fs::path(workdir) / "cache").string(), "--upload", "off", "--trust-downloads"};
				args.insert(args.end(), extra_args.begin(), extra_args.end());
				// Once a second the job's `ves` prints the numbers of its runs; they go to the host and to this worker's own display. Its other lines are kept for a failure.
				std::deque<std::string> tail;
				const int code = run_process(exe, args, [&](const std::string& text) {
					if (!text.empty() && text[0] == (char)30) {
						send(M_PROGRESS, id_bytes(id) + text.substr(1));
						ui::set_remote(rows_key, jname + ": ", ui::parse_progress(text.substr(1)));
					} else if (!text.empty()) {
						tail.push_back(text);
						if (tail.size() > 12) tail.pop_front();
					}
				});
				for (auto& l : tail) log_text += l + "\n";
				ok = code == 0;
				if (!ok) log("job " + jname + " failed with code " + std::to_string(code) + (log_text.empty() ? std::string() : ":\n" + log_text));
				// The run results are the .json files in the input's folder; the sweep summary next to it is not one.
				for (auto& folder : fs::directory_iterator(dir / "out")) {
					if (!folder.is_directory()) continue;
					for (auto& f : fs::directory_iterator(folder.path())) {
						if (f.path().extension() != ".json") continue;
						const std::string n = f.path().filename().string();
						std::string msg = id_bytes(id);
						msg += (char)(n.size() & 255);
						msg += (char)(n.size() >> 8);
						msg += n + read_file(f.path());
						if (!send(M_RESULT, msg)) break;
						sent_results++;
					}
				}
			}
			fs::remove_all(dir, ec2);
			if (ok && sent_results == 0) log("job " + jname + " made no results:\n" + log_text);
			ui::set_remote(rows_key, "", {});
			running_jobs--;
			show_title(true);
			send(M_DONE, "{\"id\":" + std::to_string(id) + ",\"ok\":" + std::string(ok ? "true" : "false") + ",\"log\":" + jq(log_text) + "}");
		};

		while (conn->recv_msg(type, payload)) {
			if (type == M_JOB) {
				auto inbox = std::make_shared<Inbox>();
				try {
					const size_t id = (size_t)atoll(Json::parse(payload).find("id")->text().c_str());
					std::lock_guard lock(inbox_m);
					inboxes[id] = inbox;
				} catch (const std::exception&) { continue; }
				jobs.emplace_back(run_job, payload, inbox);
			} else if (type == M_FILE_DATA || type == M_FILE_END) {
				std::shared_ptr<Inbox> inbox;
				{
					std::lock_guard lock(inbox_m);
					auto it = inboxes.find(id_of(payload));
					if (it != inboxes.end()) inbox = it->second;
				}
				if (inbox) inbox->push(type, payload);
			}
		}
		{
			std::lock_guard lock(inbox_m);
			for (auto& [id, inbox] : inboxes) inbox->close();
		}
		for (auto& t : jobs) t.join();
		show_title(false);
		log("lost the connection; trying again in 5 s");
		std::this_thread::sleep_for(std::chrono::seconds(5));
	}
}
