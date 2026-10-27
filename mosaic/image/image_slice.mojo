#
# image_slice.mojo
# mosaic
#
# Created by Christian Bator on 03/15/2025
#

from std.memory import Pointer
from std.algorithm import vectorize
from max.algorithm import parallelize
from std.pathlib import Path
from std.sys.info import bit_width_of

from mosaic.numeric import Matrix, MatrixSlice
from mosaic.numeric import StridedRange
from mosaic.utility import unroll_factor

from .image import Image
from .codec import ImageFile
from .color_space import ColorSpace


#
# ImageSlice
#
struct ImageSlice[mut: Bool, //, color_space: ColorSpace, dtype: DType, origin: Origin[mut=mut]](ImplicitlyCopyable, Movable, Writable):
    #
    # Fields
    #
    var _image: Pointer[Image[Self.color_space, Self.dtype], Self.origin]
    var _y_range: StridedRange
    var _x_range: StridedRange

    #
    # Initialization
    #
    def __init__(
        out self, ref[Self.origin] image: Image[Self.color_space, Self.dtype], y_range: StridedRange, x_range: StridedRange
    ) raises:
        if y_range.end > image.height() or x_range.end > image.width():
            raise Error("Out of bounds image slice for image ", image, ": y_range: ", y_range, " x_range: ", x_range)

        self._image = Pointer(to=image)
        self._y_range = y_range
        self._x_range = x_range

    def __init__(out self, existing: Self, y_range: StridedRange, x_range: StridedRange) raises:
        var new_y_range = StridedRange(
            existing._y_range.start + y_range.start * existing._y_range.step,
            existing._y_range.start + y_range.end * existing._y_range.step,
            existing._y_range.step * y_range.step,
        )

        var new_x_range = StridedRange(
            existing._x_range.start + x_range.start * existing._x_range.step,
            existing._x_range.start + x_range.end * existing._x_range.step,
            existing._x_range.step * x_range.step,
        )

        if new_y_range.end > existing._image[].height() or new_x_range.end > existing._image[].width():
            raise Error("Out of bounds image slice for image ", existing._image[], ": y_range: ", new_y_range, " x_range: ", new_x_range)

        self._image = existing._image
        self._y_range = new_y_range
        self._x_range = new_x_range

    #
    # Properties
    #
    @always_inline
    def y_range(self) -> StridedRange:
        return self._y_range

    @always_inline
    def x_range(self) -> StridedRange:
        return self._x_range

    @always_inline
    def height(self) -> Int:
        return self._y_range.count()

    @always_inline
    def width(self) -> Int:
        return self._x_range.count()

    def channels(self) -> Int:
        return Self.color_space.channels()

    #
    # Public Access
    #
    @always_inline
    def __getitem__(self, y: Int, x: Int) raises -> Scalar[Self.dtype]:
        comptime assert Self.color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"

        return self.strided_load(y=y, x=x, channel=0)

    @always_inline
    def __getitem__(self, y: Int, x: Int, channel: Int) raises -> Scalar[Self.dtype]:
        return self.strided_load(y=y, x=x, channel=channel)

    @always_inline
    def __setitem__(mut self, y: Int, x: Int, value: Scalar[Self.dtype]) raises where Self.mut:
        comptime assert Self.color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"

        self.strided_store(value, y=y, x=x, channel=0)

    @always_inline
    def __setitem__(mut self, y: Int, x: Int, channel: Int, value: Scalar[Self.dtype]) raises where Self.mut:
        self.strided_store(value, y=y, x=x, channel=channel)

    @always_inline
    def strided_load[width: Int = 1](self, y: Int, x: Int) raises -> SIMD[Self.dtype, width]:
        comptime assert Self.color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"

        return self.strided_load[width](y=y, x=x, channel=0)

    @always_inline
    def strided_load[width: Int = 1](self, y: Int, x: Int, channel: Int) raises -> SIMD[Self.dtype, width]:
        return self._image[].strided_load[width](
            y=self._y_range.start + y * self._y_range.step, x=self._x_range.start + x * self._x_range.step, channel=channel
        )

    @always_inline
    def strided_store[width: SIMDLength, //](mut self, value: SIMD[Self.dtype, width], y: Int, x: Int) raises where Self.mut:
        comptime assert Self.color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"

        self.strided_store(value, y=y, x=x, channel=0)

    @always_inline
    def strided_store[width: SIMDLength, //](mut self, value: SIMD[Self.dtype, width], y: Int, x: Int, channel: Int) raises where Self.mut:
        self._image.unsafe_mut_cast[True]()[].strided_store(
            value, y=self._y_range.start + y * self._y_range.step, x=self._x_range.start + x * self._x_range.step, channel=channel
        )

    #
    # Private Access
    #
    @always_inline
    def _strided_load[width: Int = 1](self, y: Int, x: Int) -> SIMD[Self.dtype, width]:
        comptime assert Self.color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"

        return self._strided_load[width](y=y, x=x, channel=0)

    @always_inline
    def _strided_load[width: Int = 1](self, y: Int, x: Int, channel: Int) -> SIMD[Self.dtype, width]:
        return self._image[]._strided_load[width](
            y=self._y_range.start + y * self._y_range.step, x=self._x_range.start + x * self._x_range.step, channel=channel
        )

    @always_inline
    def _strided_store[width: SIMDLength, //](mut self, value: SIMD[Self.dtype, width], y: Int, x: Int) where Self.mut:
        comptime assert Self.color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"

        self._strided_store(value, y=y, x=x, channel=0)

    @always_inline
    def _strided_store[width: SIMDLength, //](mut self, value: SIMD[Self.dtype, width], y: Int, x: Int, channel: Int) where Self.mut:
        self._image.unsafe_mut_cast[True]()[]._strided_store(
            value, y=self._y_range.start + y * self._y_range.step, x=self._x_range.start + x * self._x_range.step, channel=channel
        )

    #
    # Slicing
    #
    @always_inline
    def __getitem__(self, y: Int, x_slice: Slice) raises -> Self:
        return self[y : y + 1, x_slice]

    @always_inline
    def __getitem__(self, y_slice: Slice, x: Int) raises -> Self:
        return self[y_slice, x : x + 1]

    @always_inline
    def __getitem__(self, y_slice: Slice, x_slice: Slice) raises -> Self:
        return self.slice(
            y_range=StridedRange(
                slice=y_slice,
                default_start=0,
                default_end=self.height(),
                default_step=1,
            ),
            x_range=StridedRange(
                slice=x_slice,
                default_start=0,
                default_end=self.width(),
                default_step=1,
            ),
        )

    @always_inline
    def slice(self, y_range: StridedRange) raises -> Self:
        return self.slice(y_range=y_range, x_range=StridedRange(self.width()))

    @always_inline
    def slice(self, *, x_range: StridedRange) raises -> Self:
        return self.slice(y_range=StridedRange(self.height()), x_range=x_range)

    @always_inline
    def slice(self, y_range: StridedRange, x_range: StridedRange) raises -> Self:
        return Self(self, y_range=y_range, x_range=x_range)

    #
    # Deep Copy
    #
    def deep_copy(self) -> Image[Self.color_space, Self.dtype]:
        var result = Image[Self.color_space, Self.dtype](height=self.height(), width=self.width())

        comptime for channel in range(Self.color_space.channels()):

            def process_row(range_y: Int) {mut result, imm}:
                var y = self._y_range.start + range_y * self._y_range.step

                def process_col[width: Int](range_x: Int) {mut result, imm self, imm y, imm range_y}:
                    var x = self._x_range.start + range_x * self._x_range.step
                    result._strided_store(
                        self._image[]._strided_load[width](y=y, x=x, channel=channel), y=range_y, x=range_x, channel=channel
                    )

                vectorize[Image[Self.color_space, Self.dtype].optimal_simd_width, unroll_factor=unroll_factor](self.width(), process_col)

            parallelize(process_row, self.height())

        return result^

    #
    # Saving to File
    #
    def save[file_type: ImageFile](self, path: String) raises:
        self.deep_copy().save[file_type](path)

    def save[file_type: ImageFile](self, path: Path) raises:
        self.deep_copy().save[file_type](path)

    #
    # Writable
    #
    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "[ImageSlice: y_range = ",
            self._y_range,
            ", x_range = ",
            self._x_range,
            ", height = ",
            self.height(),
            ", width = ",
            self.width(),
            ", color_space = ",
            Self.color_space,
            ", bit_depth = ",
            bit_width_of[Self.dtype](),
            "]",
        )
