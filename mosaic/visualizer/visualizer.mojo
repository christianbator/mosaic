#
# visualizer.mojo
# mosaic
#
# Created by Christian Bator on 12/14/2024
#

from sys.ffi import _Global, _OwnedDLHandle, _get_dylib_function, c_int, c_char, c_float
from memory import UnsafePointer

from mosaic.utility import dynamic_library_filepath
from mosaic.image import Image, ImageSlice, ColorSpace
from mosaic.video import VideoCapturing

#
# Backend
#
alias _libvisualizer = _Global["libvisualizer", _OwnedDLHandle, _load_libvisualizer]()


fn _load_libvisualizer() -> _OwnedDLHandle:
    return _OwnedDLHandle(dynamic_library_filepath("libmosaic-visualizer"))


#
# Visualizer
#
struct Visualizer:
    #
    # Fields
    #
    alias display_dtype = DType.uint8

    #
    # ImageSlice
    #
    @staticmethod
    fn show[dtype: DType, color_space: ColorSpace, //](image_slice: ImageSlice[dtype, color_space], window_title: String):
        Self.show(image=image_slice.copy(), window_title=window_title)

    #
    # Image
    #
    @staticmethod
    fn show[dtype: DType, color_space: ColorSpace, //](image: Image[dtype, color_space], window_title: String):
        @parameter
        if color_space.is_display_color_space() and dtype == Self.display_dtype:
            Self._show(image=image, window_title=window_title)
        elif color_space.is_display_color_space():
            Self._show(
                image=image.as_type[Self.display_dtype](),
                window_title=window_title,
            )
        else:
            Self._show(
                image=image.converted_as_type[Self.display_dtype, ColorSpace.rgb](),
                window_title=window_title,
            )

    @staticmethod
    fn _show[dtype: DType, color_space: ColorSpace, //](image: Image[dtype, color_space], owned window_title: String):
        var show = _get_dylib_function[
            _libvisualizer,
            "show",
            fn (
                data: UnsafePointer[UInt8],
                width: c_int,
                height: c_int,
                channels: c_int,
                window_title: UnsafePointer[c_char],
            ) -> None,
        ]()

        show(
            data=image.unsafe_uint8_ptr(),
            width=c_int(image.width()),
            height=c_int(image.height()),
            channels=c_int(image.channels()),
            window_title=window_title.unsafe_cstr_ptr(),
        )

    #
    # Video
    #
    @staticmethod
    fn stream[V: VideoCapturing, //](mut video_capture: V, window_title: String):
        while True:
            if video_capture.is_next_frame_available():
                var frame = video_capture.next_frame()
                video_capture.did_read_next_frame()
                Self.show(image=frame[], window_title=window_title)

            if not Self.wait(0.001):
                break

    @staticmethod
    fn stream[
        V: VideoCapturing,
        out_dtype: DType,
        out_color_space: ColorSpace, //,
        frame_processor: fn[V: VideoCapturing] (Pointer[Image[DType.uint8, V.color_space]]) capturing [_] -> Image[out_dtype, out_color_space],
    ](mut video_capture: V, window_title: String):
        while True:
            if video_capture.is_next_frame_available():
                var frame = video_capture.next_frame()
                video_capture.did_read_next_frame()
                var processed_frame = frame_processor(frame)
                Self.show(image=processed_frame, window_title=window_title)

            if not Self.wait(0.001):
                break

    #
    # Run Loop
    #
    @staticmethod
    fn wait():
        _ = Self.wait(c_float.MAX_FINITE)

    @staticmethod
    fn wait(timeout: Float32) -> Bool:
        var wait = _get_dylib_function[_libvisualizer, "wait", fn (timeout: c_float) -> Bool]()

        return wait(c_float(timeout))
