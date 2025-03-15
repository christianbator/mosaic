#
# visualizer.mojo
# mosaic
#
# Created by Christian Bator on 12/14/2024
#

from sys.ffi import DLHandle, c_int, c_char, c_float
from memory import UnsafePointer

from mosaic.image import Image, ImagePointer, ColorSpace
from mosaic.video import VideoCapture


#
# Visualizer
#
struct Visualizer:
    alias display_dtype = DType.uint8

    @staticmethod
    fn _libvisualizer() -> DLHandle:
        return DLHandle("mosaic/libvisualizer.dylib")

    @staticmethod
    fn _close():
        var libvisualizer = Self._libvisualizer()
        libvisualizer.close()

    #
    # Image
    #
    @staticmethod
    fn show[color_space: ColorSpace, dtype: DType, //](image: Image[dtype, color_space], window_title: String):
        @parameter
        if color_space.is_display_color_space() and dtype == Self.display_dtype:
            Self._show(image=image, window_title=window_title)
        elif color_space.is_display_color_space():
            Self._show(
                image=image.astype[Self.display_dtype](),
                window_title=window_title,
            )
        else:
            Self._show(
                image=image.converted_astype[ColorSpace.rgb, Self.display_dtype](),
                window_title=window_title,
            )

    @staticmethod
    fn _show[color_space: ColorSpace, dtype: DType, //](image: Image[dtype, color_space], window_title: String):
        var show = Self._libvisualizer().get_function[
            fn (
                data: UnsafePointer[UInt8],
                width: c_int,
                height: c_int,
                channels: c_int,
                window_title: UnsafePointer[c_char],
            ) -> None
        ]("show")

        show(
            data=image.unsafe_uint8_ptr(),
            width=c_int(image.width()),
            height=c_int(image.height()),
            channels=c_int(image.channels()),
            window_title=window_title.unsafe_cstr_ptr(),
        )

        Self._close()

    #
    # Video
    #
    @staticmethod
    fn stream[VideoCaptureType: VideoCapture, //](mut video_capture: VideoCaptureType, window_title: String):
        while True:
            if video_capture.is_next_frame_available():
                var frame = video_capture.next_frame()
                video_capture.did_read_next_frame()
                Self.show(image=frame[], window_title=window_title)

            if not Self.wait(0.001):
                break

    #
    # Video (w/ Frame Processor)
    #
    @staticmethod
    fn stream[
        VideoCaptureType: VideoCapture,
        out_color_space: ColorSpace,
        out_dtype: DType, //,
        frame_processor: fn[V: VideoCapture] (ImagePointer[V.dtype, V.color_space]) capturing [_] -> ImagePointer[out_dtype, out_color_space],
    ](mut video_capture: VideoCaptureType, window_title: String):
        while True:
            if video_capture.is_next_frame_available():
                var frame = video_capture.next_frame()
                video_capture.did_read_next_frame()
                var processed_frame = frame_processor(frame)
                Self.show(image=processed_frame[], window_title=window_title)

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
        var wait = Self._libvisualizer().get_function[fn (timeout: c_float) -> Bool]("wait")

        var result = wait(c_float(timeout))
        Self._close()

        return result
