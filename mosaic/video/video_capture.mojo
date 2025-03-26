#
# video_capture.mojo
# mosaic
#
# Created by Christian Bator on 02/17/2025
#

from compile import compile_info

from memory import Pointer, UnsafePointer
from sys.ffi import DLHandle, OpaquePointer, c_int

from mosaic.image import Image, ColorSpace
from mosaic.utility import dynamic_library_filepath, fatal_error
from mosaic.numeric import Size


#
# VideoCaptureDimensions
#
@value
struct _VideoCaptureDimensions:
    var width: c_int
    var height: c_int


#
# VideoCapture
#
struct VideoCapture(VideoCapturing):
    #
    # Fields
    #
    alias dtype = DType.uint8
    alias color_space = ColorSpace.rgb

    var _video_capture: OpaquePointer
    var _frame_buffer: Image[Self.dtype, Self.color_space]
    var _is_next_frame_available: c_int

    @staticmethod
    fn _libvideocapture() -> DLHandle:
        var libvideocapture = DLHandle(dynamic_library_filepath("libmosaic-videocapture"))

        if not libvideocapture:
            fatal_error("Failed to load libmosaic-videocapture")

        return libvideocapture

    #
    # Initialization
    #
    fn __init__(out self) raises:
        var initialize = Self._libvideocapture().get_function[fn () -> OpaquePointer]("initialize")
        var video_capture = initialize()

        var open = Self._libvideocapture().get_function[fn (video_capture: OpaquePointer, dimensions: UnsafePointer[_VideoCaptureDimensions]) -> c_int]("open")

        var dimensions = _VideoCaptureDimensions(width=0, height=0)
        var result = open(video_capture, dimensions=UnsafePointer.address_of(dimensions))

        if result != 1:
            raise ("Failed to open VideoCapture")

        self._video_capture = video_capture
        self._frame_buffer = Image[Self.dtype, Self.color_space](width=Int(dimensions.width), height=Int(dimensions.height))
        self._is_next_frame_available = 1

        var info = compile_info[Self.is_next_frame_available]()
        print(info.asm)

    fn __del__(owned self):
        var deinitialize = Self._libvideocapture().get_function[fn (videoCapture: OpaquePointer) -> None]("deinitialize")
        deinitialize(self._video_capture)

    #
    # Properties
    #
    @always_inline
    fn dimensions(self) -> Size:
        return Size(width=self._frame_buffer.width(), height=self._frame_buffer.height())

    #
    # VideoCapturing
    #
    fn is_next_frame_available(self) -> Bool:
        return self._is_next_frame_available == 1

    fn next_frame(self) -> Pointer[Image[Self.dtype, Self.color_space], ImmutableAnyOrigin]:
        return rebind[Pointer[Image[Self.dtype, Self.color_space], ImmutableAnyOrigin]](Pointer.address_of(self._frame_buffer))

    fn did_read_next_frame(mut self):
        self._is_next_frame_available = 0

    #
    # Methods
    #
    fn start(mut self):
        var start = Self._libvideocapture().get_function[
            fn (video_capture: OpaquePointer, frame_buffer: UnsafePointer[UInt8], is_next_frame_available: UnsafePointer[c_int]) -> None
        ]("start")

        start(
            self._video_capture,
            frame_buffer=self._frame_buffer.unsafe_uint8_ptr(),
            is_next_frame_available=UnsafePointer.address_of(self._is_next_frame_available),
        )

    fn stop(mut self):
        var stop = Self._libvideocapture().get_function[fn (video_capture: OpaquePointer) -> None]("stop")

        stop(self._video_capture)
