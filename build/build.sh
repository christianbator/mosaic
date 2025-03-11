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
# Locations
#
build_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
artifact_dir=$build_dir/artifacts
lib_dir=$artifact_dir/lib
package_dir=$artifact_dir/package
mkdir -p $lib_dir $package_dir

#
# Build codecs
#
echo -e "> Building ${cyan}libcodec${reset} ..."

cp external/stb/{stb_image.h,stb_image_write.h} libcodec

clang -fPIC -shared -Wall -Werror -DSTBI_NEON -o $lib_dir/libcodec.dylib libcodec/libcodec.c

#
# Build visualizer
#

# Mac visualizer
echo -e "> Building ${cyan}libmac-visualizer${reset} ..."

swift_source_files=$(find mosaic/visualizer/backend/mac/MacVisualizer -name "*.swift")

swiftc -emit-library -o $lib_dir/libmac-visualizer.dylib $swift_source_files

#
# Build Mosaic
#
echo -e "> Building ${cyan}mosaic${reset} ..."

mojo package mosaic -o $package_dir/mosaic.mojopkg

#
# Build examples
#
echo -e "> Building ${cyan}examples${reset} ..."

cp $package_dir/mosaic.mojopkg examples/mosaic.mojopkg
cp -r $lib_dir examples
# mojo build examples/basic.mojo -o bin/basic

echo -e "> Build succeeded (package: ${cyan}package/mosaic.mojopkg${reset}) ${green}✔${reset}"
