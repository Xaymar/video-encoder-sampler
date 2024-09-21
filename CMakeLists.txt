cmake_minimum_required(VERSION 3.20)
project(ves LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Shared FFmpeg dev build (include/, lib/, bin/).
set(FFMPEG_DIR "${CMAKE_SOURCE_DIR}/third-party/ffmpeg-dev" CACHE PATH "FFmpeg dev build root")

set(FF_LIBS avformat avcodec avfilter avutil swscale swresample)
foreach(l ${FF_LIBS})
	find_library(FF_${l} ${l} PATHS "${FFMPEG_DIR}/lib" NO_DEFAULT_PATH REQUIRED)
	list(APPEND FF_LINK ${FF_${l}})
endforeach()

add_executable(ves src/main.cpp src/source.cpp src/vmaf.cpp src/encoder_libav.cpp src/encoder_nvenc.cpp src/encoder_amf.cpp)
target_include_directories(ves PRIVATE "${FFMPEG_DIR}/include" "${CMAKE_SOURCE_DIR}/third-party/nv-codec-headers/include" "${CMAKE_SOURCE_DIR}/third-party/amf-include")
target_link_libraries(ves PRIVATE ${FF_LINK} d3d11 dxgi)

# Runtime DLLs next to the executable.
add_custom_command(TARGET ves POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory "${FFMPEG_DIR}/bin" "$<TARGET_FILE_DIR:ves>")
