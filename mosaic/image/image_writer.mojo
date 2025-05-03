#
# image_codec.mojo
# mosaic
#
# Created by Christian Bator on 12/14/2024
#

from os import makedirs
from os.path import dirname
from pathlib import Path
from sys.ffi import _get_dylib_function, c_int, c_char
from memory import UnsafePointer

from mosaic.utility import dynamic_library_filepath, fatal_error

from .codec import _libcodec


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
    fn __init__(out self, path: Path):
        self._path = path

    #
    # Writing
    #
    fn write[dtype: DType, //, file_type: ImageFile](self, image: Image[dtype]) raises:
        var path_string: String
        if self._path.suffix() in file_type.valid_extensions():
            path_string = self._path.__fspath__()
        else:
            path_string = self._path.__fspath__() + file_type.extension()

        try:
            var dirname = dirname(self._path)
            if len(dirname) > 0:
                makedirs(path=dirname, exist_ok=True)
        except:
            raise ("Failed to create directory for image writing: " + String(self._path))

        var data: UnsafePointer[UInt8]

        @parameter
        if dtype == DType.uint8:
            data = image.unsafe_uint8_ptr()
        else:
            data = image.as_type[DType.uint8]().unsafe_uint8_ptr()

        var result: c_int

        @parameter
        if file_type == ImageFile.png:
            var write_image_data_png = _get_dylib_function[
                _libcodec,
                "write_image_data_png",
                fn (
                    filename: UnsafePointer[c_char],
                    data: UnsafePointer[UInt8],
                    width: c_int,
                    height: c_int,
                    channels: c_int,
                ) -> c_int,
            ]()

            result = write_image_data_png(
                filename=path_string.unsafe_cstr_ptr(),
                data=data,
                width=c_int(image.width()),
                height=c_int(image.height()),
                channels=c_int(image.channels()),
            )
        elif file_type == ImageFile.jpeg:
            var write_image_data_jpeg = _get_dylib_function[
                _libcodec,
                "write_image_data_jpeg",
                fn (
                    filename: UnsafePointer[c_char],
                    data: UnsafePointer[UInt8],
                    width: c_int,
                    height: c_int,
                    channels: c_int,
                ) -> c_int,
            ]()

            result = write_image_data_jpeg(
                filename=path_string.unsafe_cstr_ptr(),
                data=data,
                width=c_int(image.width()),
                height=c_int(image.height()),
                channels=c_int(image.channels()),
            )
        else:
            result = 0
            fatal_error("Unimplemented write() for image file type: ", file_type)

        if result != 1:
            raise Error("Failed to save image to file: ", self._path)
