#
# image_slice.mojo
# mosaic
#
# Created by Christian Bator on 03/15/2025
#

from math import ceildiv
from memory import Pointer
from algorithm import parallelize, vectorize

from mosaic.numeric import Matrix, MatrixSlice
from mosaic.numeric import StridedRange
from mosaic.utility import unroll_factor


#
# ImageSlice
#
@value
struct ImageSlice[mut: Bool, //, dtype: DType, color_space: ColorSpace, origin: Origin[mut]]():
    var _image: Pointer[Image[dtype, color_space], origin]
    var _y_range: StridedRange
    var _x_range: StridedRange
    var _height: Int
    var _width: Int

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

                    result.strided_store(
                        y=range_y,
                        x=range_x,
                        channel=channel,
                        value=self._image[].strided_load[width](y=y, x=x, channel=channel),
                    )

                vectorize[process_col, Image[dtype, color_space].optimal_simd_width, unroll_factor=unroll_factor](self._width)

            parallelize[process_row](self._height)

        return result^
