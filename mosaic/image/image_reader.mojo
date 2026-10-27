#
# image_reader.mojo
# mosaic
#
# Created by Christian Bator on 03/11/2025
#

from std.pathlib import Path
from std.ffi import _get_dylib_function, c_int
from std.memory.alloc import unsafe_alloc

from mosaic.utility import dynamic_library_filepath

from .codec import _libcodec
from .image import Image
from .color_space import ColorSpace


#
# ImageInfo
#
struct ImageInfo(ImplicitlyCopyable, Movable, Writable):
    var height: c_int
    var width: c_int
    var bit_depth: c_int

    def __init__(out self):
        self.height = 0
        self.width = 0
        self.bit_depth = 0

    def write_to[W: Writer](self, mut writer: W):
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
    def __init__(out self, path: Path):
        self._path = path

    #
    # Reading
    #
    def read(self) raises -> Image[Self.color_space, Self.dtype]:
        # Read raw file data
        var raw_data = self._path.read_bytes()

        # Decode image info
        var image_info = ImageInfo()

        var decode_image_info = _get_dylib_function[
            _libcodec,
            "decode_image_info",
            def(
                raw_data: Pointer[UInt8, MutUnsafeAnyOrigin],
                raw_data_length: c_int,
                image_info: Pointer[ImageInfo, MutUnsafeAnyOrigin],
            ) thin abi("C") -> c_int,
        ]()

        var is_valid_info = decode_image_info(
            raw_data=raw_data.unsafe_ptr().as_unsafe_any_origin(),
            raw_data_length=c_int(len(raw_data)),
            image_info=Pointer(to=image_info).as_unsafe_any_origin(),
        )

        if not is_valid_info:
            raise Error("Failed to read image from file (invalid info): ", self._path)

        # Decode image data
        var height = Int(image_info.height)
        var width = Int(image_info.width)
        var num_elements = height * width * Self.color_space.channels()

        var is_valid_data: c_int
        var image: Image[Self.color_space, Self.dtype]

        #
        # 8-bit images
        #
        if image_info.bit_depth == 8:
            var image_data = unsafe_alloc[UInt8](num_elements)

            var decode_image_data_uint8 = _get_dylib_function[
                _libcodec,
                "decode_image_data_uint8",
                def(
                    raw_data: Pointer[UInt8, MutUnsafeAnyOrigin],
                    raw_data_length: c_int,
                    desired_channels: c_int,
                    image_data: Pointer[UInt8, MutUnsafeAnyOrigin],
                ) thin abi("C") -> c_int,
            ]()

            is_valid_data = decode_image_data_uint8(
                raw_data=raw_data.unsafe_ptr().as_unsafe_any_origin(),
                raw_data_length=c_int(len(raw_data)),
                desired_channels=c_int(Self.color_space.channels()),
                image_data=image_data.as_unsafe_any_origin(),
            )

            comptime if Self.dtype == DType.uint8:
                image = Image[Self.color_space, Self.dtype](
                    height=height,
                    width=width,
                    data=image_data.unsafe_bitcast[Scalar[Self.dtype]](),
                )
            else:
                image = Image[Self.color_space, DType.uint8](height=height, width=width, data=image_data).as_type[Self.dtype]()

        #
        # 16-bit images
        #
        elif image_info.bit_depth == 16:
            var image_data = unsafe_alloc[UInt16](num_elements)

            var decode_image_data_uint16 = _get_dylib_function[
                _libcodec,
                "decode_image_data_uint16",
                def(
                    raw_data: Pointer[UInt8, MutUnsafeAnyOrigin],
                    raw_data_length: c_int,
                    desired_channels: c_int,
                    image_data: Pointer[UInt16, MutUnsafeAnyOrigin],
                ) thin abi("C") -> c_int,
            ]()

            is_valid_data = decode_image_data_uint16(
                raw_data=raw_data.unsafe_ptr().as_unsafe_any_origin(),
                raw_data_length=c_int(len(raw_data)),
                desired_channels=c_int(Self.color_space.channels()),
                image_data=image_data.as_unsafe_any_origin(),
            )

            comptime if Self.dtype == DType.uint16:
                image = Image[Self.color_space, Self.dtype](
                    height=height,
                    width=width,
                    data=image_data.unsafe_bitcast[Scalar[Self.dtype]](),
                )
            else:
                image = Image[Self.color_space, DType.uint16](height=height, width=width, data=image_data).as_type[Self.dtype]()

        #
        # HDR (32-bit float) images
        #
        elif image_info.bit_depth == 32:
            var image_data = unsafe_alloc[Float32](num_elements)

            var decode_image_data_float32 = _get_dylib_function[
                _libcodec,
                "decode_image_data_float32",
                def(
                    raw_data: Pointer[UInt8, MutUnsafeAnyOrigin],
                    raw_data_length: c_int,
                    desired_channels: c_int,
                    image_data: Pointer[Float32, MutUnsafeAnyOrigin],
                ) thin abi("C") -> c_int,
            ]()

            is_valid_data = decode_image_data_float32(
                raw_data=raw_data.unsafe_ptr().as_unsafe_any_origin(),
                raw_data_length=c_int(len(raw_data)),
                desired_channels=c_int(Self.color_space.channels()),
                image_data=image_data.as_unsafe_any_origin(),
            )

            comptime if Self.dtype == DType.float32:
                image = Image[Self.color_space, Self.dtype](
                    height=height,
                    width=width,
                    data=image_data.unsafe_bitcast[Scalar[Self.dtype]](),
                )
            else:
                image = Image[Self.color_space, DType.float32](height=height, width=width, data=image_data).as_type[Self.dtype]()

        #
        # Unsupported bit-depth
        #
        else:
            raise Error("Unsupported bit depth in ImageReader.read()")

        if not is_valid_data:
            raise Error("Failed to read image from file (invalid data): ", self._path)

        return image^
