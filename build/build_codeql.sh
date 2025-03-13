#
# build_codeql.sh
# mosaic
#
# Created by Christian Bator on 03/13/2025
#

#
# Locations
#
build_dir=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
artifact_dir=$build_dir/artifacts
mkdir -p $artifact_dir
lib_dir=$artifact_dir/mosaic
mkdir -p $lib_dir

#
# Build c(++) code
#
cp external/stb/{stb_image.h,stb_image_write.h} libcodec
clang -fPIC -shared -Wall -Werror -DSTBI_NEON -o $lib_dir/libcodec.dylib libcodec/libcodec.c

#
# Build swift code
#
swift_source_files=$(find libvisualizer/mac/MacVisualizer -name "*.swift")
swiftc -emit-library -o $lib_dir/libvisualizer.dylib $swift_source_files
