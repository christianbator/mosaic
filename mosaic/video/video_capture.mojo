#
# video_capture.mojo
# mosaic
#
# Created by Christian Bator on 02/17/2025
#

from memory import Pointer, UnsafePointer
from sys.ffi import _Global, _OwnedDLHandle, _get_dylib_function, OpaquePointer, c_int, c_char

from mosaic.image import Image, ColorSpace
from mosaic.utility import dynamic_library_filepath
from mosaic.numeric import Size

#
# Backend
#
alias _libvideocapture = _Global["libvideocapture", _OwnedDLHandle, _load_libvideocapture]()


fn _load_libvideocapture() -> _OwnedDLHandle:
    return _OwnedDLHandle(dynamic_library_filepath("libmosaic-videocapture"))


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
        var initialize_with_index = _get_dylib_function[_libvideocapture, "initialize_with_index", fn (index: c_int) -> OpaquePointer]()
        var video_capture = initialize_with_index(c_int(index))

        self = Self(video_capture)

    fn __init__(out self, owned name: String) raises:
        var initialize_with_name = _get_dylib_function[_libvideocapture, "initialize_with_name", fn (index: UnsafePointer[c_char]) -> OpaquePointer]()
        var video_capture = initialize_with_name(name.unsafe_cstr_ptr())

        self = Self(video_capture)

    @doc_private
    fn __init__(out self, video_capture: OpaquePointer) raises:
        # Open system video capture
        var open = _get_dylib_function[
            _libvideocapture, "open", fn (video_capture: OpaquePointer, color_space: c_int, dimensions: UnsafePointer[_VideoCaptureDimensions]) -> Bool
        ]()

        var dimensions = _VideoCaptureDimensions(width=0, height=0)

        if not open(video_capture, color_space=c_int(capture_color_space.raw_value()), dimensions=UnsafePointer.address_of(dimensions)):
            raise ("Failed to open VideoCapture")

        # Prepare properties and cache functions
        var width = Int(dimensions.width)
        var height = Int(dimensions.height)

        self._video_capture = video_capture
        self._dimensions = Size(width=width, height=height)
        self._frame_buffer = Image[DType.uint8, Self.color_space](width=width, height=height)

        self._start = _get_dylib_function[_libvideocapture, "start", fn (video_capture: OpaquePointer, frame_buffer: UnsafePointer[UInt8]) -> None]()
        self._is_next_frame_available = _get_dylib_function[_libvideocapture, "is_next_frame_available", fn (video_capture: OpaquePointer) -> Bool]()
        self._did_read_next_frame = _get_dylib_function[_libvideocapture, "did_read_next_frame", fn (video_capture: OpaquePointer) -> None]()
        self._stop = _get_dylib_function[_libvideocapture, "stop", fn (video_capture: OpaquePointer) -> None]()
        self._deinitialize = _get_dylib_function[_libvideocapture, "deinitialize", fn (videoCapture: OpaquePointer) -> None]()

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
