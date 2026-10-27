#
# image_codec.mojo
# mosaic
#
# Created by Christian Bator on 12/14/2024
#

from std.os import abort, makedirs
from std.os.path import dirname
from std.pathlib import Path
from std.ffi import _get_dylib_function, c_int, c_char

from mosaic.utility import dynamic_library_filepath

from .codec import _libcodec, ImageFile
from .image import Image
from .color_space import ColorSpace


#
# ImageWriter
#
struct ImageWriter:
    #
    # Fields
    #
    var _path: Path

    #
    # Initialization
    #
    def __init__(out self, path: Path):
        self._path = path

    #
    # Writing
    #
    def write[color_space: ColorSpace, dtype: DType, //, file_type: ImageFile](self, image: Image[color_space, dtype]) raises:
        var path_string: String
        if self._path.suffix() in file_type.valid_extensions():
            path_string = self._path.__fspath__()
        else:
            path_string = self._path.__fspath__() + file_type.extension()

        try:
            var dirname = dirname(self._path)
            if dirname.byte_length() > 0:
                makedirs(path=dirname, exist_ok=True)
        except:
            raise Error("Failed to create directory for image writing: " + String(self._path))

        # The image passed to _write() stays alive for the duration of the call, which the raw data pointer relies on
        comptime if dtype == DType.uint8:
            self._write[file_type](path_string, rebind[Image[color_space, DType.uint8]](image))
        else:
            self._write[file_type](path_string, image.as_type[DType.uint8]())

    def _write[file_type: ImageFile](self, var path_string: String, image: Image[_, DType.uint8]) raises:
        var data = image.unsafe_uint8_ptr()
        var result: c_int

        comptime if file_type == ImageFile.png:
            var write_image_data_png = _get_dylib_function[
                _libcodec,
                "write_image_data_png",
                def(
                    filename: Pointer[c_char, ImmUnsafeAnyOrigin],
                    data: Pointer[UInt8, MutUnsafeAnyOrigin],
                    height: c_int,
                    width: c_int,
                    channels: c_int,
                ) thin abi("C") -> c_int,
            ]()

            result = write_image_data_png(
                filename=path_string.as_c_string_span().ptr().as_unsafe_any_origin(),
                data=data.as_unsafe_any_origin(),
                height=c_int(image.height()),
                width=c_int(image.width()),
                channels=c_int(image.channels()),
            )
        elif file_type == ImageFile.jpeg:
            var write_image_data_jpeg = _get_dylib_function[
                _libcodec,
                "write_image_data_jpeg",
                def(
                    filename: Pointer[c_char, ImmUnsafeAnyOrigin],
                    data: Pointer[UInt8, MutUnsafeAnyOrigin],
                    height: c_int,
                    width: c_int,
                    channels: c_int,
                ) thin abi("C") -> c_int,
            ]()

            result = write_image_data_jpeg(
                filename=path_string.as_c_string_span().ptr().as_unsafe_any_origin(),
                data=data.as_unsafe_any_origin(),
                height=c_int(image.height()),
                width=c_int(image.width()),
                channels=c_int(image.channels()),
            )
        else:
            abort("Unimplemented write() for image file type: " + String(file_type))

        if result != 1:
            raise Error("Failed to save image to file: ", self._path)
