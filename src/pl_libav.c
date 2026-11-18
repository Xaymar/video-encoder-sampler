// libplacebo's FFmpeg helpers (pl_map_avframe_ex() and the rest of <libplacebo/utils/libav.h>) are written in C and compiled once, here; pl_convert.cpp
// includes the header for their declarations only.
#ifndef VES_NO_VULKAN
#include <vulkan/vulkan.h>
#define PL_LIBAV_IMPLEMENTATION 1
#include <libplacebo/utils/libav.h>
#endif
