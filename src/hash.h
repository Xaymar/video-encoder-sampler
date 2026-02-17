#pragma once
#include <string>

// SHA-256 of a file's contents as 64 lowercase hex digits; empty if the file cannot be read.
std::string sha256_file(const std::string& path);
