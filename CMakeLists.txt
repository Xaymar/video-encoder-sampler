cmake_minimum_required(VERSION 3.20)
project(ves LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(WIN32)
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
else()
	# Linux: the libraries come from the system (or the Flatpak runtime/modules) through pkg-config.
	find_package(PkgConfig REQUIRED)
	pkg_check_modules(LIBS REQUIRED IMPORTED_TARGET libavformat libavcodec libavutil x264 SvtAv1Enc libplacebo libvmaf)
	find_library(X265_LIB x265 REQUIRED)
	find_package(Threads REQUIRED)
	pkg_check_modules(VPL IMPORTED_TARGET vpl) # QuickSync headers; the library itself is loaded at run time
endif()

# libcurl, built from the submodule (git submodule update --init): static, with Windows' own TLS (Schannel) and nothing else, so that there is
# no DLL to ship. Used to download inputs given as URLs.
set(BUILD_CURL_EXE OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_LIBCURL_DOCS OFF CACHE BOOL "" FORCE)
set(BUILD_MISC_DOCS OFF CACHE BOOL "" FORCE)
set(ENABLE_CURL_MANUAL OFF CACHE BOOL "" FORCE)
if(WIN32)
	set(CURL_USE_SCHANNEL ON CACHE BOOL "" FORCE)
	set(CURL_USE_OPENSSL OFF CACHE BOOL "" FORCE)
else()
	set(CURL_USE_SCHANNEL OFF CACHE BOOL "" FORCE)
	set(CURL_USE_OPENSSL ON CACHE BOOL "" FORCE)
endif()
set(CURL_USE_LIBPSL OFF CACHE BOOL "" FORCE)
set(CURL_USE_LIBSSH2 OFF CACHE BOOL "" FORCE)
set(CURL_USE_LIBSSH OFF CACHE BOOL "" FORCE)
set(CURL_USE_LIBUV OFF CACHE BOOL "" FORCE)
set(USE_LIBIDN2 OFF CACHE BOOL "" FORCE)
set(USE_WIN32_IDN ${WIN32} CACHE BOOL "" FORCE)
set(USE_NGHTTP2 OFF CACHE BOOL "" FORCE)
set(CURL_ZLIB OFF CACHE STRING "" FORCE)
set(CURL_BROTLI OFF CACHE STRING "" FORCE)
set(CURL_ZSTD OFF CACHE STRING "" FORCE)
set(CURL_DISABLE_LDAP ON CACHE BOOL "" FORCE)
set(CURL_DISABLE_LDAPS ON CACHE BOOL "" FORCE)
set(CURL_ENABLE_EXPORT_TARGET OFF CACHE BOOL "" FORCE)
add_subdirectory(third-party/curl EXCLUDE_FROM_ALL)

add_executable(ves src/main.cpp src/source.cpp src/ff.cpp src/ff_source.cpp src/pl_convert.cpp src/ff_decode.cpp src/formats.cpp src/cache.cpp src/gpu.cpp src/versions.cpp src/ui.cpp src/pipeline.cpp src/mux.cpp src/download.cpp src/hash.cpp src/options.cpp src/upload.cpp src/priority.cpp src/vmaf.cpp src/encoder_nvenc.cpp src/encoder_amf.cpp src/encoder_qsv.cpp src/encoder_x264.cpp src/encoder_x265.cpp src/encoder_svtav1.cpp)
if(WIN32)
	target_include_directories(ves PRIVATE "${FFMPEG_DIR}/include" "${CMAKE_SOURCE_DIR}/third-party/nv-codec-headers/include" "${CMAKE_SOURCE_DIR}/third-party/amf-include" "${CMAKE_SOURCE_DIR}/third-party/libvpl/ucrt64/include" "${X264_DIR}/include" "${X265_DIR}/include" "${SVTAV1_DIR}/include" "${VMAF_DIR}/include" "${PLACEBO_DIR}/include")
	target_compile_definitions(ves PRIVATE _CRT_SECURE_NO_WARNINGS)
	target_link_libraries(ves PRIVATE ${FF_LINK} ${X264_LIB} ${X265_LIB} ${SVTAV1_LIB} ${VMAF_LIB} ${PLACEBO_LIB} d3d11 dxgi CURL::libcurl)

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
	install(FILES config.jsonc README.md DESTINATION .)
else()
	target_include_directories(ves PRIVATE "${CMAKE_SOURCE_DIR}/third-party/nv-codec-headers/include")
	target_link_libraries(ves PRIVATE PkgConfig::LIBS ${X265_LIB} Threads::Threads ${CMAKE_DL_LIBS} CURL::libcurl)
	if(VPL_FOUND)
		target_link_libraries(ves PRIVATE PkgConfig::VPL)
	endif()
	# cmake --install build --prefix /usr: the program in bin/, the VMAF models and an example config in share/ves/.
	install(TARGETS ves RUNTIME DESTINATION bin)
	install(DIRECTORY models DESTINATION share/ves)
	install(FILES config.jsonc README.md DESTINATION share/ves)
endif()
