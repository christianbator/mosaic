#
# image_reader.mojo
# mosaic
#
# Created by Christian Bator on 03/11/2025
#

from pathlib import Path
from sys.ffi import _get_dylib_function, c_int
from memory import UnsafePointer

from mosaic.utility import dynamic_library_filepath

from .codec import _libcodec


#
# ImageInfo
#
@value
struct ImageInfo(Stringable, Writable):
    var height: c_int
    var width: c_int
    var bit_depth: c_int

    fn __init__(out self):
        self.height = 0
        self.width = 0
        self.bit_depth = 0

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "[ImageInfo: height = ",
            self.height,
            ", width = ",
            self.width,
            ", bit_depth = ",
            self.bit_depth,
            "]",
        )


#
# ImageReader
#
struct ImageReader[color_space: ColorSpace, dtype: DType]:
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
    # Reading
    #
    fn read(self) raises -> Image[color_space, dtype]:
        # Read raw file data
        var raw_data = self._path.read_bytes()

        # Decode image info
        var image_info = ImageInfo()

        var decode_image_info = _get_dylib_function[
            _libcodec,
            "decode_image_info",
            fn (
                raw_data: UnsafePointer[UInt8],
                raw_data_length: c_int,
                image_info: UnsafePointer[ImageInfo],
            ) -> c_int,
        ]()

        var is_valid_info = decode_image_info(
            raw_data=raw_data.unsafe_ptr(),
            raw_data_length=len(raw_data),
            image_info=UnsafePointer.address_of(image_info),
        )

        if not is_valid_info:
            raise Error("Failed to read image from file (invalid info): ", self._path)

        # Decode image data
        var height = Int(image_info.height)
        var width = Int(image_info.width)
        var num_elements = height * width * color_space.channels()

        var is_valid_data: c_int
        var image: Image[color_space, dtype]

        #
        # 8-bit images
        #
        if image_info.bit_depth == 8:
            var image_data = UnsafePointer[UInt8].alloc(num_elements)

            var decode_image_data_uint8 = _get_dylib_function[
                _libcodec,
                "decode_image_data_uint8",
                fn (
                    raw_data: UnsafePointer[UInt8],
                    raw_data_length: c_int,
                    desired_channels: c_int,
                    image_data: UnsafePointer[UInt8],
                ) -> c_int,
            ]()

            is_valid_data = decode_image_data_uint8(
                raw_data=raw_data.unsafe_ptr(),
                raw_data_length=len(raw_data),
                desired_channels=color_space.channels(),
                image_data=image_data,
            )

            @parameter
            if dtype == DType.uint8:
                image = Image[color_space, dtype](
                    height=height,
                    width=width,
                    data=image_data.bitcast[Scalar[dtype]](),
                )
            else:
                image = Image[color_space, DType.uint8](height=height, width=width, data=image_data).as_type[dtype]()

        #
        # 16-bit images
        #
        elif image_info.bit_depth == 16:
            var image_data = UnsafePointer[UInt16].alloc(num_elements)

            var decode_image_data_uint16 = _get_dylib_function[
                _libcodec,
                "decode_image_data_uint16",
                fn (
                    raw_data: UnsafePointer[UInt8],
                    raw_data_length: c_int,
                    desired_channels: c_int,
                    image_data: UnsafePointer[UInt16],
                ) -> c_int,
            ]()

            is_valid_data = decode_image_data_uint16(
                raw_data=raw_data.unsafe_ptr(),
                raw_data_length=len(raw_data),
                desired_channels=color_space.channels(),
                image_data=image_data,
            )

            @parameter
            if dtype == DType.uint16:
                image = Image[color_space, dtype](
                    height=height,
                    width=width,
                    data=image_data.bitcast[Scalar[dtype]](),
                )
            else:
                image = Image[color_space, DType.uint16](height=height, width=width, data=image_data).as_type[dtype]()

        #
        # HDR (32-bit float) images
        #
        elif image_info.bit_depth == 32:
            var image_data = UnsafePointer[Float32].alloc(num_elements)

            var decode_image_data_float32 = _get_dylib_function[
                _libcodec,
                "decode_image_data_float32",
                fn (
                    raw_data: UnsafePointer[UInt8],
                    raw_data_length: c_int,
                    desired_channels: c_int,
                    image_data: UnsafePointer[Float32],
                ) -> c_int,
            ]()

            is_valid_data = decode_image_data_float32(
                raw_data=raw_data.unsafe_ptr(),
                raw_data_length=len(raw_data),
                desired_channels=color_space.channels(),
                image_data=image_data,
            )

            @parameter
            if dtype == DType.float32:
                image = Image[color_space, dtype](
                    height=height,
                    width=width,
                    data=image_data.bitcast[Scalar[dtype]](),
                )
            else:
                image = Image[color_space, DType.float32](height=height, width=width, data=image_data).as_type[dtype]()

        #
        # Unsupported bit-depth
        #
        else:
            raise Error("Unsupported bit depth in ImageReader.read()")

        if not is_valid_data:
            raise Error("Failed to read image from file (invalid data): ", self._path)

        return image^
