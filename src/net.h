#pragma once
// Plain TCP for distributed runs (remote.cpp): a connection that sends and receives whole messages (one type byte and a payload), and a listener.
#include <cstdint>
#include <memory>
#include <string>

namespace net {

class Conn {
public:
	explicit Conn(intptr_t fd) : fd(fd) {}
	~Conn();
	Conn(const Conn&) = delete;
	Conn& operator=(const Conn&) = delete;

	bool send_msg(uint8_t type, const std::string& payload); // false once the connection is lost
	bool recv_msg(uint8_t& type, std::string& payload);      // false once the connection is lost or the message is implausibly large
	void close();
	void shutdown();                                         // ends the connection for every thread that waits in send_msg or recv_msg, which then return false
	// A send_msg or recv_msg that makes no progress for `ms` fails (and the connection is lost); timed_out() says whether that is why the connection was lost.
	void set_timeout(int ms);
	bool timed_out() const { return timeout_hit; }
	std::string peer;                                        // "address:port" of the other side

private:
	bool send_all(const void* data, size_t size);
	bool recv_all(void* data, size_t size);
	intptr_t fd;
	bool timeout_hit = false;
};

// A connection to host:port; nullptr (and `err`) if there is none.
std::unique_ptr<Conn> connect_to(const std::string& host, int port, std::string& err);

class Listener {
public:
	~Listener();
	bool open(int port, std::string& err);
	std::unique_ptr<Conn> accept(int timeout_ms); // nullptr when nobody connected in time
	void close();

private:
	intptr_t fd = -1;
};

} // namespace net
