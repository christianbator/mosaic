#
# image_slice.mojo
# mosaic
#
# Created by Christian Bator on 03/15/2025
#

from math import ceildiv
from memory import Pointer
from algorithm import parallelize, vectorize
from pathlib import Path

from mosaic.numeric import Matrix, MatrixSlice
from mosaic.numeric import StridedRange
from mosaic.utility import unroll_factor, fatal_error


#
# ImageSlice
#
@value
struct ImageSlice[mut: Bool, //, dtype: DType, color_space: ColorSpace, origin: Origin[mut]](Stringable, Writable):
    #
    # Fields
    #
    var _image: Pointer[Image[dtype, color_space], origin]
    var _y_range: StridedRange
    var _x_range: StridedRange
    var _height: Int
    var _width: Int

    #
    # Initialization
    #
    fn __init__(
        out self,
        ref [origin]image: Image[dtype, color_space],
        y_range: StridedRange,
        x_range: StridedRange,
    ):
        self._image = Pointer.address_of(image)
        self._y_range = y_range
        self._x_range = x_range
        self._height = ceildiv(y_range.end - y_range.start, y_range.step)
        self._width = ceildiv(x_range.end - x_range.start, x_range.step)

    fn __init__(out self, other: Self, y_range: StridedRange, x_range: StridedRange):
        self._image = other._image

        self._y_range = StridedRange(
            other._y_range.start + y_range.start,
            other._y_range.start + y_range.end,
            other._y_range.step * y_range.step,
        )

        self._x_range = StridedRange(
            other._x_range.start + x_range.start,
            other._x_range.start + x_range.end,
            other._x_range.step * x_range.step,
        )

        self._height = ceildiv(y_range.end - y_range.start, y_range.step)
        self._width = ceildiv(x_range.end - x_range.start, x_range.step)

    #
    # Properties
    #
    @always_inline
    fn height(self) -> Int:
        return self._height

    @always_inline
    fn width(self) -> Int:
        return self._width

    @parameter
    fn channels(self) -> Int:
        return color_space.channels()

    #
    # Access
    #
    @always_inline
    fn __getitem__(self, y: Int, x: Int) raises -> Scalar[dtype]:
        constrained[color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"]()

        return self.strided_load[1](y=y, x=x, channel=0)

    @always_inline
    fn __getitem__(self, y: Int, x: Int, channel: Int) raises -> Scalar[dtype]:
        return self.strided_load[1](y=y, x=x, channel=channel)

    @always_inline
    fn __setitem__[origin: MutableOrigin, //](mut self: ImageSlice[dtype, _, origin], y: Int, x: Int, value: Scalar[dtype]) raises:
        constrained[color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"]()

        self.strided_store(value, y=y, x=x, channel=0)

    @always_inline
    fn __setitem__[origin: MutableOrigin, //](mut self: ImageSlice[dtype, _, origin], y: Int, x: Int, channel: Int, value: Scalar[dtype]) raises:
        self.strided_store(value, y=y, x=x, channel=channel)

    @always_inline
    fn strided_load[width: Int](self, y: Int, x: Int) raises -> SIMD[dtype, width]:
        constrained[color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"]()

        return self.strided_load[width](y=y, x=x, channel=0)

    @always_inline
    fn strided_load[width: Int](self, y: Int, x: Int, channel: Int) raises -> SIMD[dtype, width]:
        return self._image[].strided_load[width](
            y=self._y_range.start + y * self._y_range.step, x=self._x_range.start + x * self._x_range.step, channel=channel
        )

    @always_inline
    fn strided_store[origin: MutableOrigin, width: Int, //](mut self: ImageSlice[dtype, _, origin], value: SIMD[dtype, width], y: Int, x: Int) raises:
        constrained[color_space.channels() == 1, "Must specify channel for image slice in color space with channels > 1"]()

        self.strided_store(value, y=y, x=x, channel=0)

    @always_inline
    fn strided_store[
        origin: MutableOrigin, width: Int, //
    ](mut self: ImageSlice[dtype, _, origin], value: SIMD[dtype, width], y: Int, x: Int, channel: Int) raises:
        self._image[].strided_store(value, y=self._y_range.start + y * self._y_range.step, x=self._x_range.start + x * self._x_range.step, channel=channel)

    #
    # Slicing
    #
    @always_inline
    fn __getitem__(self, y_slice: Slice, x_slice: Slice) -> Self:
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
    fn slice(self, y_range: StridedRange) -> Self:
        return self.slice(y_range=y_range, x_range=StridedRange(self.width()))

    @always_inline
    fn slice(self, *, x_range: StridedRange) -> Self:
        return self.slice(y_range=StridedRange(self.height()), x_range=x_range)

    @always_inline
    fn slice(self, y_range: StridedRange, x_range: StridedRange) -> Self:
        return Self(other=self, y_range=y_range, x_range=x_range)

    #
    # Copy
    #
    fn copy(self) -> Image[dtype, color_space]:
        var result = Image[dtype, color_space](height=self._height, width=self._width)

        @parameter
        for channel in range(color_space.channels()):

            @parameter
            fn process_row(range_y: Int):
                var y = self._y_range.start + range_y * self._y_range.step

                @parameter
                fn process_col[width: Int](range_x: Int):
                    var x = self._x_range.start + range_x * self._x_range.step

                    try:
                        result.strided_store(self._image[].strided_load[width](y=y, x=x, channel=channel), y=range_y, x=range_x, channel=channel)
                    except error:
                        fatal_error(error)

                vectorize[process_col, Image[dtype, color_space].optimal_simd_width, unroll_factor=unroll_factor](self._width)

            parallelize[process_row](self._height)

        return result^

    #
    # Saving to File
    #
    fn save[file_type: ImageFileType](self, path: String) raises:
        self.copy().save[file_type](path)

    fn save[file_type: ImageFileType](self, path: Path) raises:
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
            ", width = ",
            self.width(),
            ", height = ",
            self.height(),
            ", color_space = ",
            color_space,
            ", bit_depth = ",
            dtype.bitwidth(),
            "]",
        )
