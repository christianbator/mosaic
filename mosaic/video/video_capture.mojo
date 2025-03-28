#
# video_capture.mojo
# mosaic
#
# Created by Christian Bator on 02/17/2025
#

from memory import Pointer, UnsafePointer
from sys.ffi import DLHandle, OpaquePointer, c_int, c_char

from mosaic.image import Image, ColorSpace
from mosaic.utility import dynamic_library_filepath, fatal_error
from mosaic.numeric import Size


#
# VideoCaptureDimensions
#
@value
struct _VideoCaptureDimensions:
    var height: c_int
    var width: c_int


#
# VideoCapture
#
struct VideoCapture[capture_color_space: ColorSpace](VideoCapturing):
    #
    # Fields
    #
    alias color_space = capture_color_space

    var _video_capture: OpaquePointer
    var _dimensions: Size
    var _frame_buffer: Image[DType.uint8, capture_color_space]

    var _start: fn (video_capture: OpaquePointer, frame_buffer: UnsafePointer[UInt8]) -> None
    var _is_next_frame_available: fn (video_capture: OpaquePointer) -> Bool
    var _did_read_next_frame: fn (video_capture: OpaquePointer) -> None
    var _stop: fn (video_capture: OpaquePointer) -> None
    var _deinitialize: fn (videoCapture: OpaquePointer) -> None

    #
    # Initialization
    #
    fn __init__(out self, index: Int) raises:
        # Load libvideocapture
        var libvideocapture = DLHandle(dynamic_library_filepath("libmosaic-videocapture"))

        # Initialize system video capture
        var initialize_with_index = libvideocapture.get_function[fn (index: c_int) -> OpaquePointer]("initialize_with_index")
        var video_capture = initialize_with_index(c_int(index))

        self = Self(video_capture, libvideocapture)

    fn __init__(out self, name: String) raises:
        # Load libvideocapture
        var libvideocapture = DLHandle(dynamic_library_filepath("libmosaic-videocapture"))

        # Initialize system video capture
        var initialize_with_name = libvideocapture.get_function[fn (index: UnsafePointer[c_char]) -> OpaquePointer]("initialize_with_name")
        var video_capture = initialize_with_name(name.unsafe_cstr_ptr())

        self = Self(video_capture, libvideocapture)

    @doc_private
    fn __init__(out self, video_capture: OpaquePointer, libvideocapture: DLHandle) raises:
        # Open system video capture
        var open = libvideocapture.get_function[
            fn (video_capture: OpaquePointer, color_space: c_int, dimensions: UnsafePointer[_VideoCaptureDimensions]) -> Bool
        ]("open")
        var dimensions = _VideoCaptureDimensions(width=0, height=0)

        if not open(video_capture, color_space=c_int(capture_color_space.raw_value()), dimensions=UnsafePointer.address_of(dimensions)):
            raise ("Failed to open VideoCapture")

        # Prepare properties and cache functions
        var width = Int(dimensions.width)
        var height = Int(dimensions.height)

        self._video_capture = video_capture
        self._dimensions = Size(width=width, height=height)
        self._frame_buffer = Image[DType.uint8, Self.color_space](width=width, height=height)

        self._start = libvideocapture.get_function[fn (video_capture: OpaquePointer, frame_buffer: UnsafePointer[UInt8]) -> None]("start")
        self._is_next_frame_available = libvideocapture.get_function[fn (video_capture: OpaquePointer) -> Bool]("is_next_frame_available")
        self._did_read_next_frame = libvideocapture.get_function[fn (video_capture: OpaquePointer) -> None]("did_read_next_frame")
        self._stop = libvideocapture.get_function[fn (video_capture: OpaquePointer) -> None]("stop")
        self._deinitialize = libvideocapture.get_function[fn (videoCapture: OpaquePointer) -> None]("deinitialize")

    fn __del__(owned self):
        self._deinitialize(self._video_capture)

    #
    # Properties
    #
    @always_inline
    fn dimensions(self) -> Size:
        return self._dimensions

    #
    # Methods
    #
    fn start(mut self):
        self._start(self._video_capture, frame_buffer=self._frame_buffer.unsafe_uint8_ptr())

    fn stop(mut self):
        self._stop(self._video_capture)

    #
    # VideoCapturing
    #
    fn is_next_frame_available(self) -> Bool:
        return self._is_next_frame_available(self._video_capture)

    fn next_frame(self) -> Pointer[Image[DType.uint8, capture_color_space], ImmutableAnyOrigin]:
        return rebind[Pointer[Image[DType.uint8, capture_color_space], ImmutableAnyOrigin]](Pointer.address_of(self._frame_buffer))

    fn did_read_next_frame(mut self):
        self._did_read_next_frame(self._video_capture)
