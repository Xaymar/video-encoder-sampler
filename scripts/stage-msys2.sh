#!/usr/bin/env bash
# Copies the MSYS2 UCRT64 libraries VES builds against into third-party/<name>/ucrt64/{include,lib,bin}, the layout CMakeLists.txt expects.
# Run it in an MSYS2 UCRT64 shell after installing the packages:
#   pacman -S --needed mingw-w64-ucrt-x86_64-{libx264,x265,svt-av1,vmaf,libplacebo,libvpl,lcms2,libdovi,shaderc,spirv-cross,vulkan-headers,libgcc,libstdc++,libwinpthread}
# Usage: scripts/stage-msys2.sh [third-party directory]
#
# Only the named packages are copied, not their whole dependency tree: the libraries' DLLs have to sit next to ves.exe, nothing else does.
# (STAGE_PACMAN and MSYS_ROOT let a test point this at a private pacman root instead of the real one.)
set -euo pipefail

dest=${1:-third-party}
root=${MSYS_ROOT:-}
pacman_cmd=${STAGE_PACMAN:-pacman}
prefix=mingw-w64-ucrt-x86_64-

declare -A groups=(
	[x264]="libx264"
	[x265]="x265 libgcc libstdc++ libwinpthread"
	[svt-av1]="svt-av1"
	[vmaf]="vmaf"
	[libplacebo]="libplacebo lcms2 libdovi shaderc spirv-cross vulkan-headers"
	[libvpl]="libvpl"
)

for name in "${!groups[@]}"; do
	out="$dest/$name"
	rm -rf "$out"
	mkdir -p "$out"
	for pkg in ${groups[$name]}; do
		$pacman_cmd -Ql "$prefix$pkg" | while read -r _ path; do
			path=${path#"$root"} # (a private pacman root is shown in front of the path)
			case "$path" in
				*/) continue ;;
				/ucrt64/include/* | /ucrt64/lib/*.a | /ucrt64/bin/*.dll) ;;
				*) continue ;;
			esac
			[[ "$path" == */vulkan-1.dll ]] && continue # the GPU driver's loader is the one to use
			mkdir -p "$out/$(dirname "${path#/}")"
			cp "$root$path" "$out/${path#/}"
		done
	done
	echo "$name: $(find "$out" -type f | wc -l) files, DLLs: $(ls "$out/ucrt64/bin" 2>/dev/null | tr '\n' ' ')"
done
