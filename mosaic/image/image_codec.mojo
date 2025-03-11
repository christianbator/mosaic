#
# image_codec.mojo
# mosaic
#
# Created by Christian Bator on 12/14/2024
#

from os import abort, makedirs
from os.path import dirname
from pathlib import Path
from sys.ffi import DLHandle, c_int, c_char
from memory import UnsafePointer

from mosaic.matrix import Matrix

#
# Backend
#
var _libcodec = DLHandle("lib/libcodec.dylib")

var decode_image_info = _libcodec.get_function[
    fn (raw_data: UnsafePointer[UInt8], raw_data_length: c_int, image_info: UnsafePointer[ImageInfo]) -> c_int
]("decode_image_info")

var _decode_image_data_uint8 = _libcodec.get_function[
    fn (
        raw_data: UnsafePointer[UInt8],
        raw_data_length: c_int,
        desired_channels: c_int,
        image_data: UnsafePointer[UInt8]
    ) -> c_int
]("decode_image_data_uint8")

var _decode_image_data_uint16 = _libcodec.get_function[
    fn (
        raw_data: UnsafePointer[UInt8],
        raw_data_length: c_int,
        desired_channels: c_int,
        image_data: UnsafePointer[UInt16]
    ) -> c_int
]("decode_image_data_uint16")

var _decode_image_data_float32 = _libcodec.get_function[
    fn (
        raw_data: UnsafePointer[UInt8],
        raw_data_length: c_int,
        desired_channels: c_int,
        image_data: UnsafePointer[Float32]
    ) -> c_int
]("decode_image_data_float32")

var _write_image_data_png = _libcodec.get_function[
    fn (
        filename: UnsafePointer[c_char],
        data: UnsafePointer[UInt8],
        width: c_int,
        height: c_int,
        channels: c_int
    ) -> c_int
]("write_image_data_png")

var _write_image_data_jpeg = _libcodec.get_function[
    fn (
        filename: UnsafePointer[c_char],
        data: UnsafePointer[UInt8],
        width: c_int,
        height: c_int,
        channels: c_int
    ) -> c_int
]("write_image_data_jpeg")

#
# ImageInfo
#
@value
struct ImageInfo(Stringable, Writable):

    var width: c_int
    var height: c_int
    var bit_depth: c_int

    fn __init__(out self):
        self.width = 0
        self.height = 0
        self.bit_depth = 0

    fn __str__(self) -> String:
        return "[ImageInfo: width = " + String(self.width) + ", height = " + String(self.height) + ", bit_depth = " + String(self.bit_depth) + "]"

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(String(self))

#
# ImageFileType
#
@value
struct ImageFileType(EqualityComparable, Stringable, Writable):

    alias png = Self(ImageFileType._png)
    alias jpeg = Self(ImageFileType._jpeg)

    alias _png = String("png")
    alias _jpeg = String("jpeg")

    alias _supported_image_file_types = [Self._png, Self._jpeg]

    var _raw_value: String

    fn __init__(out self, raw_value: String):
        self._raw_value = raw_value

        if raw_value not in Self._supported_image_file_types:
            abort("Unsupported image file type: ", raw_value)

    fn extension(self) -> String:
        if self == Self.png:
            return ".png"
        elif self == Self.jpeg:
            return ".jpeg"
        else:
            abort("Unimplemented extension() for image file type: ", self._raw_value)
            while True:
                pass

    fn __eq__(self, other: Self) -> Bool:
        return self._raw_value == other._raw_value

    fn __ne__(self, other: Self) -> Bool:
        return not(self == other)
    
    fn __str__(self) -> String:
        return "[ImageFileType: " + self._raw_value + "]"
    
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(String(self))

#
# ImageReader
#
struct ImageReader[color_space: ColorSpace, dtype: DType]:
    
    var _path: Path

    fn __init__(out self, path: Path):
        self._path = path

    fn read(self) raises -> Matrix[dtype, color_space.channels()]:
        # Read raw file data
        var raw_data = self._path.read_bytes()

        # Decode image info
        var image_info = ImageInfo()

        var is_valid_info = decode_image_info(
            raw_data = raw_data.unsafe_ptr(),
            raw_data_length = len(raw_data),
            image_info = UnsafePointer.address_of(image_info)
        )

        if not is_valid_info:
            raise Error("Failed to read image from file (invalid info): ", self._path)

        # Decode image data
        var width = Int(image_info.width)
        var height = Int(image_info.height)
        var num_elements = height * width * color_space.channels()

        var is_valid_data: c_int
        var matrix: Matrix[dtype, color_space.channels()]

        #
        # 8-bit images
        #
        if image_info.bit_depth == 8:
            var image_data = UnsafePointer[UInt8].alloc(num_elements)

            is_valid_data = _decode_image_data_uint8(
                raw_data = raw_data.unsafe_ptr(),
                raw_data_length = len(raw_data),
                desired_channels = color_space.channels(),
                image_data = image_data
            )

            @parameter
            if dtype == DType.uint8:
                matrix = Matrix[dtype, color_space.channels()](rows = height, cols = width, data = image_data.bitcast[Scalar[dtype]]())
            else:
                matrix = Matrix[DType.uint8, color_space.channels()](rows = height, cols = width, data = image_data).astype[dtype]()

        #
        # 16-bit images
        #
        elif image_info.bit_depth == 16:
            var image_data = UnsafePointer[UInt16].alloc(num_elements)

            is_valid_data = _decode_image_data_uint16(
                raw_data = raw_data.unsafe_ptr(),
                raw_data_length = len(raw_data),
                desired_channels = color_space.channels(),
                image_data = image_data
            )

            @parameter
            if dtype == DType.uint16:
                matrix = Matrix[dtype, color_space.channels()](rows = height, cols = width, data = image_data.bitcast[Scalar[dtype]]())
            else:
                matrix = Matrix[DType.uint16, color_space.channels()](rows = height, cols = width, data = image_data).astype[dtype]()

        #
        # HDR (32-bit float) images
        #
        elif image_info.bit_depth == 32:
            var image_data = UnsafePointer[Float32].alloc(num_elements)

            is_valid_data = _decode_image_data_float32(
                raw_data = raw_data.unsafe_ptr(),
                raw_data_length = len(raw_data),
                desired_channels = color_space.channels(),
                image_data = image_data
            )

            @parameter
            if dtype == DType.float32:
                matrix = Matrix[dtype, color_space.channels()](rows = height, cols = width, data = image_data.bitcast[Scalar[dtype]]())
            else:
                matrix = Matrix[DType.float32, color_space.channels()](rows = height, cols = width, data = image_data).astype[dtype]()
        
        #
        # Unsupported bit-depth
        #
        else:
            abort()
            
            # Bypass the pass manager
            is_valid_data = 0
            matrix = Matrix[dtype, color_space.channels()](rows = height, cols = width)
            ##

        if not is_valid_data:
            raise Error("Failed to read image from file (invalid data): ", self._path)

        return matrix^

#
# ImageWriter
#
struct ImageWriter:

    var _path: Path

    fn __init__(out self, path: Path):
        self._path = path

    fn write[dtype: DType, //, file_type: ImageFileType](self, image: Image[dtype, _]) raises:
        var path_string = self._path.__fspath__()
        if not path_string.endswith(file_type.extension()):
            raise Error("Mismatched file type and extension " + String(file_type) + ": " + path_string)

        makedirs(path = dirname(self._path), exist_ok = True)
        
        var data: UnsafePointer[UInt8]
        @parameter
        if dtype == DType.uint8:
            data = image.unsafe_uint8_ptr()
        else:
            data = image.astype[DType.uint8]().unsafe_uint8_ptr()
        
        var result: c_int
        @parameter
        if file_type == ImageFileType.png:
            result = _write_image_data_png(
                filename = path_string.unsafe_cstr_ptr(),
                data = data,
                width = c_int(image.width()),
                height = c_int(image.height()),
                channels = c_int(image.channels()),
            )
        elif file_type == ImageFileType.jpeg:
            result = _write_image_data_jpeg(
                filename = path_string.unsafe_cstr_ptr(),
                data = data,
                width = c_int(image.width()),
                height = c_int(image.height()),
                channels = c_int(image.channels()),
            )
        else:
            result = 0
            abort("Unimplemented write() for image file type: ", file_type)

        if result != 1:
            raise Error("Failed to save image to file: ", self._path)
