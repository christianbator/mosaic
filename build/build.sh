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
os=$(uname)

#
# Locations
#
build_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
artifact_dir=$build_dir/artifacts
mkdir -p $artifact_dir
lib_dir=$artifact_dir/mosaic
mkdir -p $lib_dir

#
# Build libcodec
#
echo -e "> Building ${cyan}libcodec${reset} ..."

if [[ "$os" == "Darwin" ]]; then
    clang -fPIC -shared -Wall -Werror -DSTBI_NEON -o $lib_dir/libcodec.dylib libcodec/libcodec.c
else
    clang -fPIC -shared -Wall -Werror -o $lib_dir/libcodec.so libcodec/libcodec.c
fi

#
# Build libvisualizer
#
echo -e "> Building ${cyan}libvisualizer${reset} ..."

if [[ "$os" == "Darwin" ]]; then
    swift_source_files=$(find libvisualizer/mac/MacVisualizer -name "*.swift")
    swiftc -emit-library -o $lib_dir/libvisualizer.dylib $swift_source_files
else
    echo -e "> Unsupported platform for libvisualizer: ${bright_red}$os${reset}"
fi

#
# Build mosaic
#
echo -e "> Building ${cyan}mosaic${reset} ..."

mojo package mosaic -o $artifact_dir/mosaic.mojopkg

echo -e "> Build succeeded (package: ${cyan}package/mosaic.mojopkg${reset}) ${green}✔${reset}"
