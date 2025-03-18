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
mojo_package_dir=$lib_dir/mojo

mkdir -p $lib_dir
mkdir -p $mojo_package_dir

#
# Build libmosaic-codec
#
echo -e "> Building ${cyan}libmosaic-codec${reset} ..."

stb_options=""

if [[ "$os" == "macOS" ]]; then
    additional_stb_options="-DSTBI_NEON"
fi

clang -fPIC -shared -Wall -Werror $stb_options -o $lib_dir/libmosaic-codec$SHLIB_EXT libmosaic-codec/codec.c

#
# Build libmosaic-visualizer
#
echo -e "> Building ${cyan}libmosaic-visualizer${reset} ..."

if [[ "$os" == "macOS" ]]; then
    swift_source_files=$(find libmosaic-visualizer/mac/MacVisualizer -name "*.swift")
    swiftc -emit-library -o $lib_dir/libmosaic-visualizer$SHLIB_EXT $swift_source_files
else
    echo -e "  > ${bright_red}Warning:${reset} Unsupported os for libmosaic-visualizer: $os, skipping ..."
fi

#
# Build mosaic
#
echo -e "> Building ${cyan}mosaic.mojopkg${reset} ..."

mojo package mosaic -o $mojo_package_dir/mosaic.mojopkg

echo -e "> Build succeeded ${green}✔${reset}"
