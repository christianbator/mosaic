#
# build.sh
# mosaic
#
# Created by Christian Bator on 01/02/2025
#

set -euo pipefail

#
# Locations
#
mkdir -p $PREFIX/lib/mojo
mkdir -p $PREFIX/lib/mosaic

#
# Build libcodec
#
cp external/stb/{stb_image.h,stb_image_write.h} libcodec

if [ -n "$OSX_ARCH" ]; then
    clang -fPIC -shared -Wall -Werror -DSTBI_NEON -o $PREFIX/lib/mosaic/libcodec.dylib libcodec/libcodec.c
else
    clang -fPIC -shared -Wall -Werror -o $PREFIX/lib/mosaic/libcodec.dylib libcodec/libcodec.c
fi

#
# Build libvisualizer
#
if [ -n "$OSX_ARCH" ]; then
    swift_source_files=$(find libvisualizer/mac/MacVisualizer -name "*.swift")
    swiftc -emit-library -o $PREFIX/lib/mosaic/libvisualizer.dylib $swift_source_files
else
    echo "> Unsupported platform for libvisualizer"
fi

#
# Build mosaic
#
mojo package mosaic -o $PREFIX/lib/mojo/mosaic.mojopkg
