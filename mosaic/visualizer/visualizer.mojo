#
# visualizer.mojo
# mosaic
#
# Created by Christian Bator on 12/14/2024
#

from std.os import abort
from std.ffi import _Global, OwnedDLHandle, _get_dylib_function, c_int, c_char, c_float

from mosaic.utility import dynamic_library_filepath
from mosaic.image import Image, ImageSlice, ColorSpace

#
# Backend
#
comptime _libvisualizer = _Global["libvisualizer", _load_libvisualizer]()


def _load_libvisualizer() -> OwnedDLHandle:
    try:
        return OwnedDLHandle(dynamic_library_filepath("libmosaic-visualizer"))
    except:
        abort()


#
# Visualizer
#
struct Visualizer:
    #
    # Fields
    #
    comptime display_dtype = DType.uint8

    #
    # ImageSlice
    #
    @staticmethod
    def show(image_slice: ImageSlice, window_title: String):
        Self.show(image=image_slice.deep_copy(), window_title=window_title)

    #
    # Image
    #
    @staticmethod
    def show[dtype: DType, color_space: ColorSpace, //](image: Image[color_space, dtype], window_title: String):
        comptime if color_space.is_display_color_space() and dtype == Self.display_dtype:
            Self._show(image=image, window_title=window_title)
        elif color_space.is_display_color_space():
            Self._show(
                image=image.as_type[Self.display_dtype](),
                window_title=window_title,
            )
        else:
            Self._show(
                image=image.converted_as_type[ColorSpace.rgb, Self.display_dtype](),
                window_title=window_title,
            )

    @staticmethod
    def _show[dtype: DType, color_space: ColorSpace, //](image: Image[color_space, dtype], var window_title: String):
        try:
            var show = _get_dylib_function[
                _libvisualizer,
                "show",
                def(
                    data: Pointer[UInt8, MutUnsafeAnyOrigin],
                    height: c_int,
                    width: c_int,
                    channels: c_int,
                    window_title: Pointer[c_char, ImmUnsafeAnyOrigin],
                ) thin abi("C") -> None,
            ]()

            show(
                data=image.unsafe_uint8_ptr().as_unsafe_any_origin(),
                height=c_int(image.height()),
                width=c_int(image.width()),
                channels=c_int(image.channels()),
                window_title=window_title.as_c_string_span().ptr().as_unsafe_any_origin(),
            )
        except:
            abort()

    #
    # Run Loop
    #
    @staticmethod
    def wait():
        _ = Self.wait(c_float.MAX_FINITE)

    @staticmethod
    def wait(timeout: Float32) -> Bool:
        try:
            var wait = _get_dylib_function[_libvisualizer, "wait", def(timeout: c_float) thin abi("C") -> Bool]()

            return wait(c_float(timeout))
        except:
            abort()
