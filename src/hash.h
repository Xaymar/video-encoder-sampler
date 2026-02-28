#pragma once
#include <string>

// SHA-256 as 64 lowercase hex digits: of a file's contents (empty if the file cannot be read), or of a string's bytes.
std::string sha256_file(const std::string& path);
std::string sha256_bytes(const std::string& data);
