#
# image_slice.mojo
# mosaic
#
# Created by Christian Bator on 03/15/2025
#

from memory import Pointer
from algorithm import parallelize, vectorize
from pathlib import Path

from mosaic.numeric import Matrix, MatrixSlice
from mosaic.numeric import StridedRange
from mosaic.utility import unroll_factor


#
# ImageSlice
#
@value
struct ImageSlice[mut: Bool, //, color_space: ColorSpace, dtype: DType, origin: Origin[mut]](Stringable, Writable):
    #
    # Fields
    #
    var _image: Pointer[Image[color_space, dtype], origin]
    var _y_range: StridedRange
    var _x_range: StridedRange

    #
    # Initialization
    #
    fn __init__(out self, ref [origin]image: Image[color_space, dtype], y_range: StridedRange, x_range: StridedRange) raises:
        if y_range.end > image.height() or x_range.end > image.width():
            raise Error("Out of bounds image slice for image ", image, ": y_range: ", y_range, " x_range: ", x_range)

        self._image = Pointer.address_of(image)
        self._y_range = y_range
        self._x_range = x_range

    fn __init__(out self, existing: Self, y_range: StridedRange, x_range: StridedRange) raises:
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
    fn y_range(self) -> StridedRange:
        return self._y_range

    @always_inline
    fn x_range(self) -> StridedRange:
        return self._x_range

    @always_inline
    fn height(self) -> Int:
        return self._y_range.count()

    @always_inline
    fn width(self) -> Int:
        return self._x_range.count()

    @parameter
    fn channels(self) -> Int:
        return color_space.channels()

    #
    # Public Access
    #
    @always_inline
    fn __getitem__(self, y: Int, x: Int) raises -> Scalar[dtype]:
        constrained[color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"]()

        return self.strided_load(y=y, x=x, channel=0)

    @always_inline
    fn __getitem__(self, y: Int, x: Int, channel: Int) raises -> Scalar[dtype]:
        return self.strided_load(y=y, x=x, channel=channel)

    @always_inline
    fn __setitem__[origin: MutableOrigin, //](mut self: ImageSlice[_, dtype, origin], y: Int, x: Int, value: Scalar[dtype]) raises:
        constrained[color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"]()

        self.strided_store(value, y=y, x=x, channel=0)

    @always_inline
    fn __setitem__[origin: MutableOrigin, //](mut self: ImageSlice[_, dtype, origin], y: Int, x: Int, channel: Int, value: Scalar[dtype]) raises:
        self.strided_store(value, y=y, x=x, channel=channel)

    @always_inline
    fn strided_load[width: Int = 1](self, y: Int, x: Int) raises -> SIMD[dtype, width]:
        constrained[color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"]()

        return self.strided_load[width](y=y, x=x, channel=0)

    @always_inline
    fn strided_load[width: Int = 1](self, y: Int, x: Int, channel: Int) raises -> SIMD[dtype, width]:
        return self._image[].strided_load[width](
            y=self._y_range.start + y * self._y_range.step, x=self._x_range.start + x * self._x_range.step, channel=channel
        )

    @always_inline
    fn strided_store[origin: MutableOrigin, width: Int, //](mut self: ImageSlice[_, dtype, origin], value: SIMD[dtype, width], y: Int, x: Int) raises:
        constrained[color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"]()

        self.strided_store(value, y=y, x=x, channel=0)

    @always_inline
    fn strided_store[
        origin: MutableOrigin, width: Int, //
    ](mut self: ImageSlice[_, dtype, origin], value: SIMD[dtype, width], y: Int, x: Int, channel: Int) raises:
        self._image[].strided_store(value, y=self._y_range.start + y * self._y_range.step, x=self._x_range.start + x * self._x_range.step, channel=channel)

    #
    # Private Access
    #
    @always_inline
    fn _strided_load[width: Int = 1](self, y: Int, x: Int) -> SIMD[dtype, width]:
        constrained[color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"]()

        return self._strided_load[width](y=y, x=x, channel=0)

    @always_inline
    fn _strided_load[width: Int = 1](self, y: Int, x: Int, channel: Int) -> SIMD[dtype, width]:
        return self._image[]._strided_load[width](
            y=self._y_range.start + y * self._y_range.step, x=self._x_range.start + x * self._x_range.step, channel=channel
        )

    @always_inline
    fn _strided_store[origin: MutableOrigin, width: Int, //](mut self: ImageSlice[_, dtype, origin], value: SIMD[dtype, width], y: Int, x: Int):
        constrained[color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"]()

        self._strided_store(value, y=y, x=x, channel=0)

    @always_inline
    fn _strided_store[origin: MutableOrigin, width: Int, //](mut self: ImageSlice[_, dtype, origin], value: SIMD[dtype, width], y: Int, x: Int, channel: Int):
        self._image[]._strided_store(value, y=self._y_range.start + y * self._y_range.step, x=self._x_range.start + x * self._x_range.step, channel=channel)

    #
    # Slicing
    #
    @always_inline
    fn __getitem__(self, y: Int, x_slice: Slice) raises -> Self:
        return self[y : y + 1, x_slice]

    @always_inline
    fn __getitem__(self, y_slice: Slice, x: Int) raises -> Self:
        return self[y_slice, x : x + 1]

    @always_inline
    fn __getitem__(self, y_slice: Slice, x_slice: Slice) raises -> Self:
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
    fn slice(self, y_range: StridedRange) raises -> Self:
        return self.slice(y_range=y_range, x_range=StridedRange(self.width()))

    @always_inline
    fn slice(self, *, x_range: StridedRange) raises -> Self:
        return self.slice(y_range=StridedRange(self.height()), x_range=x_range)

    @always_inline
    fn slice(self, y_range: StridedRange, x_range: StridedRange) raises -> Self:
        return Self(self, y_range=y_range, x_range=x_range)

    #
    # Copy
    #
    fn copy(self) -> Image[color_space, dtype]:
        var result = Image[color_space, dtype](height=self.height(), width=self.width())

        @parameter
        for channel in range(color_space.channels()):

            @parameter
            fn process_row(range_y: Int):
                var y = self._y_range.start + range_y * self._y_range.step

                @parameter
                fn process_col[width: Int](range_x: Int):
                    var x = self._x_range.start + range_x * self._x_range.step
                    result._strided_store(self._image[]._strided_load[width](y=y, x=x, channel=channel), y=range_y, x=range_x, channel=channel)

                vectorize[process_col, Image[color_space, dtype].optimal_simd_width, unroll_factor=unroll_factor](self.width())

            parallelize[process_row](self.height())

        return result^

    #
    # Saving to File
    #
    fn save[file_type: ImageFile](self, path: String) raises:
        self.copy().save[file_type](path)

    fn save[file_type: ImageFile](self, path: Path) raises:
        self.copy().save[file_type](path)

    #
    # Stringable & Writable
    #
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
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
            color_space,
            ", bit_depth = ",
            dtype.bitwidth(),
            "]",
        )
