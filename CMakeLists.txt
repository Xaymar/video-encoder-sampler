cmake_minimum_required(VERSION 3.20)
project(ves LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Shared FFmpeg dev build (include/, lib/, bin/).
set(FFMPEG_DIR "${CMAKE_SOURCE_DIR}/third-party/ffmpeg-dev" CACHE PATH "FFmpeg dev build root")

set(FF_LIBS avformat avcodec avutil)
foreach(l ${FF_LIBS})
	find_library(FF_${l} ${l} PATHS "${FFMPEG_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
	list(APPEND FF_LINK ${FF_${l}})
endforeach()

# libx264 (MSYS2 ucrt64 package): headers, import lib and DLL.
set(X264_DIR "${CMAKE_SOURCE_DIR}/third-party/x264/ucrt64" CACHE PATH "libx264 package root")
find_library(X264_LIB x264.dll PATHS "${X264_DIR}/lib" NO_DEFAULT_PATH REQUIRED)

# libx265 (MSYS2 ucrt64 package plus the MinGW C++ runtime DLLs it needs).
set(X265_DIR "${CMAKE_SOURCE_DIR}/third-party/x265/ucrt64" CACHE PATH "libx265 package root")
find_library(X265_LIB x265.dll PATHS "${X265_DIR}/lib" NO_DEFAULT_PATH REQUIRED)

# SVT-AV1 (MSYS2 ucrt64 package).
set(SVTAV1_DIR "${CMAKE_SOURCE_DIR}/third-party/svt-av1/ucrt64" CACHE PATH "SVT-AV1 package root")
find_library(SVTAV1_LIB SvtAv1Enc.dll PATHS "${SVTAV1_DIR}/lib" NO_DEFAULT_PATH REQUIRED)

# libplacebo (MSYS2 ucrt64 packages: libplacebo with lcms2, libdovi, shaderc and spirv-cross).
set(PLACEBO_DIR "${CMAKE_SOURCE_DIR}/third-party/libplacebo/ucrt64" CACHE PATH "libplacebo package root")
find_library(PLACEBO_LIB placebo.dll PATHS "${PLACEBO_DIR}/lib" NO_DEFAULT_PATH REQUIRED)

# libvmaf (MSYS2 ucrt64 package).
set(VMAF_DIR "${CMAKE_SOURCE_DIR}/third-party/vmaf/ucrt64" CACHE PATH "libvmaf package root")
find_library(VMAF_LIB vmaf.dll PATHS "${VMAF_DIR}/lib" NO_DEFAULT_PATH REQUIRED)

add_executable(ves src/main.cpp src/source.cpp src/ff.cpp src/ff_source.cpp src/pl_convert.cpp src/ff_decode.cpp src/formats.cpp src/cache.cpp src/gpu.cpp src/versions.cpp src/vmaf.cpp src/encoder_nvenc.cpp src/encoder_amf.cpp src/encoder_qsv.cpp src/encoder_x264.cpp src/encoder_x265.cpp src/encoder_svtav1.cpp)
target_include_directories(ves PRIVATE "${FFMPEG_DIR}/include" "${CMAKE_SOURCE_DIR}/third-party/nv-codec-headers/include" "${CMAKE_SOURCE_DIR}/third-party/amf-include" "${CMAKE_SOURCE_DIR}/third-party/libvpl/ucrt64/include" "${X264_DIR}/include" "${X265_DIR}/include" "${SVTAV1_DIR}/include" "${VMAF_DIR}/include" "${PLACEBO_DIR}/include")
target_compile_definitions(ves PRIVATE _CRT_SECURE_NO_WARNINGS)
target_link_libraries(ves PRIVATE ${FF_LINK} ${X264_LIB} ${X265_LIB} ${SVTAV1_LIB} ${VMAF_LIB} ${PLACEBO_LIB} d3d11 dxgi)

# Runtime DLLs next to the executable (only the DLLs: the packages' bin folders also hold command line tools).
file(GLOB RUNTIME_DLLS CONFIGURE_DEPENDS
	"${FFMPEG_DIR}/bin/*.dll" "${X264_DIR}/bin/*.dll" "${X265_DIR}/bin/*.dll" "${SVTAV1_DIR}/bin/*.dll"
	"${CMAKE_SOURCE_DIR}/third-party/libvpl/ucrt64/bin/libvpl-2.dll" "${VMAF_DIR}/bin/libvmaf.dll" "${PLACEBO_DIR}/bin/*.dll")
add_custom_command(TARGET ves POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different ${RUNTIME_DLLS} "$<TARGET_FILE_DIR:ves>")

# cmake --install build --prefix <dir>: the program, its DLLs, the VMAF models and an example config.
install(TARGETS ves RUNTIME DESTINATION .)
install(FILES ${RUNTIME_DLLS} DESTINATION .)
install(DIRECTORY models DESTINATION .)
install(FILES config.json README.md DESTINATION .)
