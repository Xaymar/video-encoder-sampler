#!/bin/sh
# Compiles the compute shaders to SPIR-V (the .spv files are committed and embedded into ves by CMake, so building ves needs no shader compiler).
# Needs glslc (the Vulkan SDK, the shaderc package, or third-party/libplacebo/ucrt64/bin/glslc.exe). Run from anywhere: sh shaders/build.sh [path/to/glslc]
cd "$(dirname "$0")" || exit 1
GLSLC=${1:-glslc}
for f in *.comp; do
	"$GLSLC" --target-env=vulkan1.2 -O -I . -o "${f%.comp}.spv" "$f" || exit 1
done
