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
if [ $(uname) == "Darwin" ]; then
    os="macOS"
    dynamic_lib_extension="dylib"
else
    os="linux"
    dynamic_lib_extension="so"
fi

#
# Locations
#
lib_dir=$PREFIX/lib
mojo_package_dir=$lib_dir/mojo

mkdir -p $lib_dir
mkdir -p $mojo_package_dir

#
# Check for build sub-commands
#
build_libmosaic_codec=false
build_libmosaic_visualizer=false
build_mosaic=false

if [ $# -eq 0 ]; then
    build_libmosaic_codec=true
    build_libmosaic_visualizer=true
    build_mosaic=true
else
    for arg in $@; do
        if [ $arg == "libmosaic-codec" ]; then
            build_libmosaic_codec=true
        elif [ $arg == "libmosaic-visualizer" ]; then
            build_libmosaic_visualizer=true
        elif [ $arg == "mosaic" ]; then
            build_mosaic=true
        else
            echo -e "> ${bright_red}[Error]${reset} Unrecognized build command: $arg"
            exit 1
        fi
    done
fi

#
# Build libmosaic-codec
#
if $build_libmosaic_codec; then
    echo -e "> Building ${cyan}libmosaic-codec${reset} ..."

    stb_options=""

    if [ $os == "macOS" ]; then
        additional_stb_options="-DSTBI_NEON"
    fi

    clang -fPIC -shared -Wall -Werror $stb_options -o $lib_dir/libmosaic-codec.$dynamic_lib_extension libmosaic-codec/codec.c
fi

#
# Build libmosaic-visualizer
#
if $build_libmosaic_visualizer; then
    echo -e "> Building ${cyan}libmosaic-visualizer${reset} ..."

    if [ $os == "macOS" ]; then
        swift_source_files=$(find libmosaic-visualizer/mac/MacVisualizer -name "*.swift")
        swiftc -emit-library -o $lib_dir/libmosaic-visualizer.$dynamic_lib_extension $swift_source_files
    else
        echo -e "  > ${bright_red}[Warning]${reset} Unsupported os for libmosaic-visualizer: $os, skipping ..."
    fi
fi

#
# Build mosaic
#
if $build_mosaic; then
    echo -e "> Building ${cyan}mosaic${reset} ..."

    mojo package mosaic -o $mojo_package_dir/mosaic.mojopkg
fi

echo -e "> Build succeeded ${green}✔${reset}"
