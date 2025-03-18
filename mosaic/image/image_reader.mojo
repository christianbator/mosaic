#
# image_reader.mojo
# mosaic
#
# Created by Christian Bator on 03/11/2025
#

from pathlib import Path
from sys.ffi import DLHandle, c_int
from memory import UnsafePointer

from mosaic.utility import dynamic_library_filepath, fatal_error


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
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(
            "[ImageInfo: width = ",
            self.width,
            ", height = ",
            self.height,
            ", bit_depth = ",
            self.bit_depth,
            "]",
        )


#
# ImageReader
#
struct ImageReader[dtype: DType, color_space: ColorSpace]:
    #
    # Fields
    #
    var _path: Path

    @staticmethod
    fn _libcodec() -> DLHandle:
        var libcodec = DLHandle(dynamic_library_filepath("libmosaic-codec"))

        if not libcodec:
            fatal_error("Failed to load libmosaic-codec")

        return libcodec

    #
    # Initialization
    #
    fn __init__(out self, path: Path):
        self._path = path

    #
    # Reading
    #
    fn read(self) raises -> Image[dtype, color_space]:
        var libcodec = Self._libcodec()

        # Read raw file data
        var raw_data = self._path.read_bytes()

        # Decode image info
        var image_info = ImageInfo()

        var decode_image_info = libcodec.get_function[
            fn (
                raw_data: UnsafePointer[UInt8],
                raw_data_length: c_int,
                image_info: UnsafePointer[ImageInfo],
            ) -> c_int
        ]("decode_image_info")

        var is_valid_info = decode_image_info(
            raw_data=raw_data.unsafe_ptr(),
            raw_data_length=len(raw_data),
            image_info=UnsafePointer.address_of(image_info),
        )

        if not is_valid_info:
            raise Error("Failed to read image from file (invalid info): ", self._path)

        # Decode image data
        var width = Int(image_info.width)
        var height = Int(image_info.height)
        var num_elements = height * width * color_space.channels()

        var is_valid_data: c_int
        var image: Image[dtype, color_space]

        #
        # 8-bit images
        #
        if image_info.bit_depth == 8:
            var image_data = UnsafePointer[UInt8].alloc(num_elements)

            var decode_image_data_uint8 = libcodec.get_function[
                fn (
                    raw_data: UnsafePointer[UInt8],
                    raw_data_length: c_int,
                    desired_channels: c_int,
                    image_data: UnsafePointer[UInt8],
                ) -> c_int
            ]("decode_image_data_uint8")

            is_valid_data = decode_image_data_uint8(
                raw_data=raw_data.unsafe_ptr(),
                raw_data_length=len(raw_data),
                desired_channels=color_space.channels(),
                image_data=image_data,
            )

            @parameter
            if dtype == DType.uint8:
                image = Image[dtype, color_space](
                    width=width,
                    height=height,
                    data=image_data.bitcast[Scalar[dtype]](),
                )
            else:
                image = Image[DType.uint8, color_space](width=width, height=height, data=image_data).astype[dtype]()

        #
        # 16-bit images
        #
        elif image_info.bit_depth == 16:
            var image_data = UnsafePointer[UInt16].alloc(num_elements)

            var decode_image_data_uint16 = libcodec.get_function[
                fn (
                    raw_data: UnsafePointer[UInt8],
                    raw_data_length: c_int,
                    desired_channels: c_int,
                    image_data: UnsafePointer[UInt16],
                ) -> c_int
            ]("decode_image_data_uint16")

            is_valid_data = decode_image_data_uint16(
                raw_data=raw_data.unsafe_ptr(),
                raw_data_length=len(raw_data),
                desired_channels=color_space.channels(),
                image_data=image_data,
            )

            @parameter
            if dtype == DType.uint16:
                image = Image[dtype, color_space](
                    width=width,
                    height=height,
                    data=image_data.bitcast[Scalar[dtype]](),
                )
            else:
                image = Image[DType.uint16, color_space](width=width, height=height, data=image_data).astype[dtype]()

        #
        # HDR (32-bit float) images
        #
        elif image_info.bit_depth == 32:
            var image_data = UnsafePointer[Float32].alloc(num_elements)

            var decode_image_data_float32 = libcodec.get_function[
                fn (
                    raw_data: UnsafePointer[UInt8],
                    raw_data_length: c_int,
                    desired_channels: c_int,
                    image_data: UnsafePointer[Float32],
                ) -> c_int
            ]("decode_image_data_float32")

            is_valid_data = decode_image_data_float32(
                raw_data=raw_data.unsafe_ptr(),
                raw_data_length=len(raw_data),
                desired_channels=color_space.channels(),
                image_data=image_data,
            )

            @parameter
            if dtype == DType.float32:
                image = Image[dtype, color_space](
                    width=width,
                    height=height,
                    data=image_data.bitcast[Scalar[dtype]](),
                )
            else:
                image = Image[DType.float32, color_space](width=width, height=height, data=image_data).astype[dtype]()

        #
        # Unsupported bit-depth
        #
        else:
            raise Error("Unsupported bit depth in ImageReader.read()")

        if not is_valid_data:
            raise Error("Failed to read image from file (invalid data): ", self._path)

        return image^
