#pragma once
// The compute shaders (shaders/*.spv), embedded into the program by CMake (see CMakeLists.txt): SPIR-V by name, e.g. "motion_y"; nullptr if there is none.
#include <cstddef>
const unsigned char* spv_find(const char* name, size_t& size);
