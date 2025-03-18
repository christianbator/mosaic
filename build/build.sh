#!/usr/bin/env bash

#
# build.sh
# mosaic
#
# Created by Christian Bator on 01/02/2025
#

set -euo pipefail

#
# Colors
#
cyan="\033[36m"
green="\033[32m"
bright_red="\033[91m"
reset="\033[0m"

#
# System Info
#
if [[ "$(uname)" == "Darwin" ]]; then
    os="macOS"
else
    os="linux"
fi

#
# Locations
#
lib_dir=$PREFIX/lib

mojo_lib_dir=$PREFIX/lib/mojo
mosaic_lib_dir=$PREFIX/lib/mosaic

mkdir -p $mojo_lib_dir
mkdir -p $mosaic_lib_dir

#
# Build libcodec
#
echo -e "> Building ${cyan}libcodec${reset} ..."

stb_options=""

if [[ "$os" == "macOS" ]]; then
    additional_stb_options="-DSTBI_NEON"
fi

clang -fPIC -shared -Wall -Werror $stb_options -o $mosaic_lib_dir/libcodec$SHLIB_EXT libcodec/libcodec.c

#
# Build libvisualizer
#
echo -e "> Building ${cyan}libvisualizer${reset} ..."

if [[ "$os" == "macOS" ]]; then
    swift_source_files=$(find libvisualizer/mac/MacVisualizer -name "*.swift")
    swiftc -emit-library -o $mosaic_lib_dir/libvisualizer$SHLIB_EXT $swift_source_files
else
    echo -e "  > ${bright_red}Warning:${reset} Unsupported os for libvisualizer: ${cyan}$os${reset}, skipping ..."
fi

#
# Build mosaic
#
echo -e "> Building ${cyan}mosaic${reset} ..."

mojo package mosaic -o $mojo_lib_dir/mosaic.mojopkg

echo -e "> Build succeeded: ${cyan}mosaic.mojopkg${reset} ${green}✔${reset}"
