#
# image.mojo
# mosaic
#
# Created by Christian Bator on 12/14/2024
#

from pathlib import Path
from memory import UnsafePointer
from algorithm import vectorize, parallelize
from collections import Optional
from math import floor, ceil, ceildiv, trunc, Ceilable, CeilDivable, Floorable, Truncable
from bit import next_power_of_two

from mosaic.numeric import Matrix, MatrixSlice, StridedRange, SIMDRange, Number
from mosaic.utility import optimal_simd_width, unroll_factor, fatal_error

from .image_reader import ImageReader
from .image_writer import ImageWriter
from .filters import Filters


#
# Image
#
struct Image[dtype: DType, color_space: ColorSpace](
    Absable, Ceilable, CeilDivable, EqualityComparable, ExplicitlyCopyable, Floorable, Movable, Roundable, Stringable, Truncable, Writable
):
    #
    # Fields
    #
    alias optimal_simd_width = optimal_simd_width[dtype]()

    var _matrix: Matrix[dtype, color_space.channels()]

    #
    # Initialization
    #
    fn __init__(out self, path: StringLiteral) raises:
        self = Self(Path(path))

    fn __init__(out self, path: Path) raises:
        self = ImageReader[dtype, color_space](path).read()

    fn __init__(out self, width: Int, height: Int):
        self._matrix = Matrix[dtype, color_space.channels()](rows=height, cols=width)

    # This is an unsafe convenience constructor
    fn __init__(out self, width: Int, height: Int, owned data: UnsafePointer[Scalar[dtype]]):
        self._matrix = Matrix[dtype, color_space.channels()](rows=height, cols=width, data=data)

    fn __init__(out self, owned matrix: Matrix[dtype, color_space.channels()]):
        self._matrix = matrix^

    fn __moveinit__(out self, owned existing: Self):
        self._matrix = existing._matrix^

    @staticmethod
    fn with_single_channel_data[channel: Int](single_channel_matrix: Matrix[dtype]) -> Self:
        return Self(single_channel_matrix.copied_to_component[channel, color_space.channels()]())

    @staticmethod
    fn from_spectrum[spectrum_dtype: DType, //](spectrum: Matrix[spectrum_dtype, color_space.channels(), complex=True]) -> Self:
        return Self(spectrum.fourier_transform[inverse=True]().real[dtype]())

    #
    # Properties
    #
    @always_inline
    fn width(self) -> Int:
        return self._matrix.cols()

    @always_inline
    fn height(self) -> Int:
        return self._matrix.rows()

    @always_inline
    fn pixels(self) -> Int:
        return self._matrix.strided_count()

    @parameter
    fn channels(self) -> Int:
        return color_space.channels()

    @always_inline
    fn samples(self) -> Int:
        return self._matrix.count()

    @parameter
    fn bit_depth(self) -> Int:
        return dtype.bitwidth()

    @always_inline
    fn bytes(self) -> Int:
        return self.samples() * self.bit_depth() // 8

    @always_inline
    fn matrix(self) -> ref [__origin_of(self._matrix)] Matrix[dtype, color_space.channels()]:
        return self._matrix

    #
    # Public Access
    #
    @always_inline
    fn __getitem__(self, y: Int, x: Int) raises -> Scalar[dtype]:
        constrained[color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"]()

        return self[y, x, 0]

    @always_inline
    fn __getitem__(self, y: Int, x: Int, channel: Int) raises -> Scalar[dtype]:
        return self._matrix[y, x, channel].value

    @always_inline
    fn __setitem__(mut self, y: Int, x: Int, value: Scalar[dtype]) raises:
        constrained[color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"]()

        self[y, x, 0] = value

    @always_inline
    fn __setitem__(mut self, y: Int, x: Int, channel: Int, value: Scalar[dtype]) raises:
        self._matrix[y, x, channel] = value

    @always_inline
    fn strided_load[width: Int](self, y: Int, x: Int) raises -> SIMD[dtype, width]:
        constrained[color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"]()

        return self.strided_load[width](y=y, x=x, channel=0)

    @always_inline
    fn strided_load[width: Int](self, y: Int, x: Int, channel: Int) raises -> SIMD[dtype, width]:
        return self._matrix.strided_load[width](row=y, col=x, component=channel).value

    @always_inline
    fn strided_store[width: Int](mut self, value: SIMD[dtype, width], y: Int, x: Int) raises:
        constrained[color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"]()

        self.strided_store(value, y=y, x=x, channel=0)

    @always_inline
    fn strided_store[width: Int](mut self, value: SIMD[dtype, width], y: Int, x: Int, channel: Int) raises:
        self._matrix.strided_store[width](value, row=y, col=x, component=channel)

    #
    # Unsafe Access
    #
    @always_inline
    fn unsafe_data_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self._matrix.unsafe_data_ptr()

    @always_inline
    fn unsafe_uint8_ptr(self) -> UnsafePointer[UInt8]:
        return self._matrix.unsafe_uint8_ptr()

    #
    # Slicing
    #
    @always_inline
    fn __getitem__[mut: Bool, origin: Origin[mut], //](ref [origin]self, y: Int, x_slice: Slice) raises -> ImageSlice[dtype, color_space, origin]:
        return self[y : y + 1, x_slice]

    @always_inline
    fn __getitem__[mut: Bool, origin: Origin[mut], //](ref [origin]self, y_slice: Slice, x: Int) raises -> ImageSlice[dtype, color_space, origin]:
        return self[y_slice, x : x + 1]

    @always_inline
    fn __getitem__[mut: Bool, origin: Origin[mut], //](ref [origin]self, y_slice: Slice, x_slice: Slice) raises -> ImageSlice[dtype, color_space, origin]:
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
    fn slice[mut: Bool, origin: Origin[mut], //](ref [origin]self, y_range: StridedRange) raises -> ImageSlice[dtype, color_space, origin]:
        return self.slice(y_range=y_range, x_range=StridedRange(self.width()))

    @always_inline
    fn slice[mut: Bool, origin: Origin[mut], //](ref [origin]self, *, x_range: StridedRange) raises -> ImageSlice[dtype, color_space, origin]:
        return self.slice(y_range=StridedRange(self.height()), x_range=x_range)

    @always_inline
    fn slice[
        mut: Bool, origin: Origin[mut], //
    ](ref [origin]self, y_range: StridedRange, x_range: StridedRange) raises -> ImageSlice[dtype, color_space, origin]:
        return ImageSlice[dtype, color_space, origin](image=self, y_range=y_range, x_range=x_range)

    @always_inline
    fn channel_slice[
        channel: Int
    ](self) raises -> MatrixSlice[StridedRange(channel, channel + 1), dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.component_slice[channel]()

    @always_inline
    fn channel_slice[
        channel: Int
    ](self, y_range: StridedRange) raises -> MatrixSlice[StridedRange(channel, channel + 1), dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.component_slice[channel](row_range=y_range)

    @always_inline
    fn channel_slice[
        channel: Int
    ](self, *, x_range: StridedRange) raises -> MatrixSlice[
        StridedRange(channel, channel + 1), dtype, color_space.channels(), False, __origin_of(self._matrix)
    ]:
        return self._matrix.component_slice[channel](col_range=x_range)

    @always_inline
    fn channel_slice[
        channel: Int
    ](self, y_range: StridedRange, x_range: StridedRange) raises -> MatrixSlice[
        StridedRange(channel, channel + 1), dtype, color_space.channels(), False, __origin_of(self._matrix)
    ]:
        return self._matrix.component_slice[channel](row_range=y_range, col_range=x_range)

    @always_inline
    fn strided_slice[channel_range: StridedRange](self) raises -> MatrixSlice[channel_range, dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.strided_slice[channel_range]()

    @always_inline
    fn strided_slice[
        channel_range: StridedRange
    ](self, y_range: StridedRange) raises -> MatrixSlice[channel_range, dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.strided_slice[channel_range](row_range=y_range)

    @always_inline
    fn strided_slice[
        channel_range: StridedRange
    ](self, *, x_range: StridedRange) raises -> MatrixSlice[channel_range, dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.strided_slice[channel_range](col_range=x_range)

    @always_inline
    fn strided_slice[
        channel_range: StridedRange
    ](self, y_range: StridedRange, x_range: StridedRange) raises -> MatrixSlice[channel_range, dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.strided_slice[channel_range](y_range, x_range)

    @always_inline
    fn extract_channel[channel: Int](self) -> Matrix[dtype]:
        return self._matrix.extract_component[channel]()

    fn store_sub_image(mut self, value: Self, y: Int, x: Int) raises:
        self.store_sub_image(value[:, :], y=y, x=x)

    fn store_sub_image(mut self, value: ImageSlice[dtype=dtype, color_space=color_space], y: Int, x: Int) raises:
        if (value.y_range().end > self.height()) or (value.x_range().end > self.width()):
            raise Error("Attempt to store sub-image out of bounds")

        @parameter
        for channel in range(color_space.channels()):

            @parameter
            fn store_row(sub_y: Int):
                @parameter
                fn store_cols[width: Int](sub_x: Int):
                    try:
                        self.strided_store(value.strided_load[width](y=sub_y, x=sub_x, channel=channel), y=y + sub_y, x=x + sub_x, channel=channel)
                    except error:
                        fatal_error(error)

                vectorize[store_cols, Self.optimal_simd_width, unroll_factor=unroll_factor](value.width())

            parallelize[store_row](value.height())

    fn store_sub_matrix(mut self, value: Matrix[dtype, color_space.channels()], y: Int, x: Int) raises:
        self._matrix.store_sub_matrix(value, row=y, col=x)

    fn store_sub_matrix(mut self, value: MatrixSlice[dtype=dtype, depth = color_space.channels(), complex=False], y: Int, x: Int) raises:
        self._matrix.store_sub_matrix(value, row=y, col=x)

    #
    # ExplicitlyCopyable
    #
    fn copy(self) -> Self:
        return Self(self._matrix.copy())

    fn copy_into(self, mut other: Self):
        self._matrix.copy_into(other._matrix)

    #
    # EqualityComparable
    #
    fn __eq__(self, other: Self) -> Bool:
        return self._matrix == other._matrix

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    #
    # Operators (Scalar)
    #
    fn __neg__(self) -> Self:
        return self * -1

    fn __add__(self, rhs: Scalar[dtype]) -> Self:
        return Self(self._matrix + rhs)

    fn __iadd__(mut self, rhs: Scalar[dtype]):
        self._matrix += rhs

    fn __sub__(self, rhs: Scalar[dtype]) -> Self:
        return Self(self._matrix - rhs)

    fn __isub__(mut self, rhs: Scalar[dtype]):
        self._matrix -= rhs

    fn __mul__(self, rhs: Scalar[dtype]) -> Self:
        return Self(self._matrix * rhs)

    fn __rmul__(self, lhs: Scalar[dtype]) -> Self:
        return self * lhs

    fn __imul__(mut self, rhs: Scalar[dtype]):
        self._matrix *= rhs

    fn __truediv__(self, rhs: Scalar[dtype]) -> Self:
        return Self(self._matrix / rhs)

    fn __itruediv__(mut self, rhs: Scalar[dtype]):
        self._matrix /= rhs

    fn __floordiv__(self, rhs: Scalar[dtype]) -> Self:
        return Self(self._matrix // rhs)

    fn __ifloordiv__(mut self, rhs: Scalar[dtype]):
        self._matrix //= rhs

    fn __mod__(self, rhs: Scalar[dtype]) -> Self:
        return Self(self._matrix % rhs)

    fn __imod__(mut self, rhs: Scalar[dtype]):
        self._matrix %= rhs

    fn __pow__(self, rhs: Scalar[dtype]) -> Self:
        return Self(self._matrix**rhs)

    fn __ipow__(mut self, rhs: Scalar[dtype]):
        self._matrix **= rhs

    fn __and__(self, rhs: Scalar[dtype]) -> Self:
        return Self(self._matrix & rhs)

    fn __iand__(mut self, rhs: Scalar[dtype]):
        self._matrix &= rhs

    fn __or__(self, rhs: Scalar[dtype]) -> Self:
        return Self(self._matrix | rhs)

    fn __ior__(mut self, rhs: Scalar[dtype]):
        self._matrix |= rhs

    fn __xor__(self, rhs: Scalar[dtype]) -> Self:
        return Self(self._matrix ^ rhs)

    fn __ixor__(mut self, rhs: Scalar[dtype]):
        self._matrix ^= rhs

    fn __lshift__(self, rhs: Scalar[dtype]) -> Self:
        return Self(self._matrix << rhs)

    fn __ilshift__(mut self, rhs: Scalar[dtype]):
        self._matrix <<= rhs

    fn __rshift__(self, rhs: Scalar[dtype]) -> Self:
        return Self(self._matrix >> rhs)

    fn __irshift__(mut self, rhs: Scalar[dtype]):
        self._matrix >>= rhs

    #
    # Operators (Matrix)
    #
    fn __add__(self, rhs: Matrix[dtype, color_space.channels()]) -> Self:
        return Self(self._matrix + rhs)

    fn __iadd__(mut self, rhs: Matrix[dtype, color_space.channels()]):
        self._matrix += rhs

    fn __sub__(self, rhs: Matrix[dtype, color_space.channels()]) -> Self:
        return Self(self._matrix - rhs)

    fn __isub__(mut self, rhs: Matrix[dtype, color_space.channels()]):
        self._matrix -= rhs

    fn __truediv__(self, rhs: Matrix[dtype, color_space.channels()]) -> Self:
        return Self(self._matrix / rhs)

    fn __itruediv__(mut self, rhs: Matrix[dtype, color_space.channels()]):
        self._matrix /= rhs

    fn __floordiv__(self, rhs: Matrix[dtype, color_space.channels()]) -> Self:
        return Self(self._matrix // rhs)

    fn __ifloordiv__(mut self, rhs: Matrix[dtype, color_space.channels()]):
        self._matrix //= rhs

    fn __mod__(self, rhs: Matrix[dtype, color_space.channels()]) -> Self:
        return Self(self._matrix % rhs)

    fn __imod__(mut self, rhs: Matrix[dtype, color_space.channels()]):
        self._matrix %= rhs

    fn __pow__(self, rhs: Matrix[dtype, color_space.channels()]) -> Self:
        return Self(self._matrix**rhs)

    fn __ipow__(mut self, rhs: Matrix[dtype, color_space.channels()]):
        self._matrix **= rhs

    fn __and__(self, rhs: Matrix[dtype, color_space.channels()]) -> Self:
        return Self(self._matrix & rhs)

    fn __iand__(mut self, rhs: Matrix[dtype, color_space.channels()]):
        self._matrix &= rhs

    fn __or__(self, rhs: Matrix[dtype, color_space.channels()]) -> Self:
        return Self(self._matrix | rhs)

    fn __ior__(mut self, rhs: Matrix[dtype, color_space.channels()]):
        self._matrix |= rhs

    fn __xor__(self, rhs: Matrix[dtype, color_space.channels()]) -> Self:
        return Self(self._matrix ^ rhs)

    fn __ixor__(mut self, rhs: Matrix[dtype, color_space.channels()]):
        self._matrix ^= rhs

    fn __lshift__(self, rhs: Matrix[dtype, color_space.channels()]) -> Self:
        return Self(self._matrix << rhs)

    fn __ilshift__(mut self, rhs: Matrix[dtype, color_space.channels()]):
        self._matrix <<= rhs

    fn __rshift__(self, rhs: Matrix[dtype, color_space.channels()]) -> Self:
        return Self(self._matrix >> rhs)

    fn __irshift__(mut self, rhs: Matrix[dtype, color_space.channels()]):
        self._matrix >>= rhs

    #
    # Operators (Image)
    #
    fn __add__(self, rhs: Self) -> Self:
        return Self(self._matrix + rhs._matrix)

    fn __iadd__(mut self, rhs: Self):
        self._matrix += rhs._matrix

    fn __sub__(self, rhs: Self) -> Self:
        return Self(self._matrix - rhs._matrix)

    fn __isub__(mut self, rhs: Self):
        self._matrix -= rhs._matrix

    fn __truediv__(self, rhs: Self) -> Self:
        return Self(self._matrix / rhs._matrix)

    fn __itruediv__(mut self, rhs: Self):
        self._matrix /= rhs._matrix

    fn __floordiv__(self, rhs: Self) -> Self:
        return Self(self._matrix // rhs._matrix)

    fn __ifloordiv__(mut self, rhs: Self):
        self._matrix //= rhs._matrix

    fn __mod__(self, rhs: Self) -> Self:
        return Self(self._matrix % rhs._matrix)

    fn __imod__(mut self, rhs: Self):
        self._matrix %= rhs._matrix

    fn __pow__(self, rhs: Self) -> Self:
        return Self(self._matrix**rhs._matrix)

    fn __ipow__(mut self, rhs: Self):
        self._matrix **= rhs._matrix

    fn __and__(self, rhs: Self) -> Self:
        return Self(self._matrix & rhs._matrix)

    fn __iand__(mut self, rhs: Self):
        self._matrix &= rhs._matrix

    fn __or__(self, rhs: Self) -> Self:
        return Self(self._matrix | rhs._matrix)

    fn __ior__(mut self, rhs: Self):
        self._matrix |= rhs._matrix

    fn __xor__(self, rhs: Self) -> Self:
        return Self(self._matrix ^ rhs._matrix)

    fn __ixor__(mut self, rhs: Self):
        self._matrix ^= rhs._matrix

    fn __lshift__(self, rhs: Self) -> Self:
        return Self(self._matrix << rhs._matrix)

    fn __ilshift__(mut self, rhs: Self):
        self._matrix <<= rhs._matrix

    fn __rshift__(self, rhs: Self) -> Self:
        return Self(self._matrix >> rhs._matrix)

    fn __irshift__(mut self, rhs: Self):
        self._matrix >>= rhs._matrix

    #
    # Numeric Traits
    #
    fn __floor__(self) -> Self:
        return Self(floor(self._matrix))

    fn __ceil__(self) -> Self:
        return Self(ceil(self._matrix))

    fn __ceildiv__(self, rhs: Self) -> Self:
        return Self(ceildiv(self._matrix, rhs._matrix))

    fn __round__(self) -> Self:
        return Self(round(self._matrix))

    fn __round__(self, ndigits: Int) -> Self:
        return Self(round(self._matrix, ndigits=ndigits))

    fn __trunc__(self) -> Self:
        return Self(trunc(self._matrix))

    fn __abs__(self) -> Self:
        return Self(abs(self._matrix))

    #
    # Numeric Methods
    #
    fn log(self) -> Self:
        return Self(self._matrix.log())

    fn clamp(mut self, lower_bound: Scalar[dtype], upper_bound: Scalar[dtype]):
        self._matrix.clamp(lower_bound=lower_bound, upper_bound=upper_bound)

    fn clamped(self, lower_bound: Scalar[dtype], upper_bound: Scalar[dtype]) -> Self:
        return Self(self._matrix.clamped(lower_bound=lower_bound, upper_bound=upper_bound))

    fn strided_sum(self, channel: Int) -> Scalar[dtype]:
        return self._matrix.strided_sum(channel).value

    fn strided_average(self, channel: Int) -> Scalar[DType.float64]:
        return self._matrix.strided_average(channel).value

    fn strided_min(self, channel: Int) -> Scalar[dtype]:
        return self._matrix.strided_min(channel).value

    fn strided_max(self, channel: Int) -> Scalar[dtype]:
        return self._matrix.strided_max(channel).value

    fn strided_normalize(mut self):
        self._matrix.strided_normalize()

    fn strided_fill(mut self, value: Scalar[dtype], channel: Int):
        self._matrix.strided_fill(value, component=channel)

    fn fill(mut self, value: Scalar[dtype]):
        self._matrix.fill(value)

    fn strided_for_each[transformer: fn[width: Int] (value: SIMD[dtype, width]) capturing -> SIMD[dtype, width]](mut self, channel: Int):
        @parameter
        fn number_transformer[width: Int](value: Number[dtype, width]) -> Number[dtype, width]:
            return Number[dtype, width](transformer(value.value))

        self._matrix.strided_for_each[number_transformer](channel)

    fn strided_for_each_zipped[
        transformer: fn[width: Int] (value: SIMD[dtype, width], rhs: SIMD[dtype, width]) capturing -> SIMD[dtype, width]
    ](mut self, other: Self, channel: Int):
        @parameter
        fn number_transformer[width: Int](value: Number[dtype, width], rhs: Number[dtype, width]) -> Number[dtype, width]:
            return Number[dtype, width](transformer(value=value.value, rhs=rhs.value))

        self._matrix.strided_for_each_zipped[number_transformer](other._matrix, component=channel)

    fn for_each[transformer: fn[width: Int] (value: SIMD[dtype, width]) capturing -> SIMD[dtype, width]](mut self):
        @parameter
        fn number_transformer[width: Int](value: Number[dtype, width]) -> Number[dtype, width]:
            return Number[dtype, width](transformer(value.value))

        self._matrix.for_each[number_transformer]()

    fn for_each_zipped[transformer: fn[width: Int] (value: SIMD[dtype, width], rhs: SIMD[dtype, width]) capturing -> SIMD[dtype, width]](mut self, other: Self):
        @parameter
        fn number_transformer[width: Int](value: Number[dtype, width], rhs: Number[dtype, width]) -> Number[dtype, width]:
            return Number[dtype, width](transformer(value=value.value, rhs=rhs.value))

        self._matrix.for_each_zipped[number_transformer](other._matrix)

    fn invert(mut self):
        constrained[dtype == DType.uint8, "invert() is only available for UInt8 images, since inversion is ambiguous otherwise"]()

        @parameter
        fn transformer[width: Int](value: SIMD[dtype, width]) -> SIMD[dtype, width]:
            return 255 - value

        self.for_each[transformer]()

    fn inverted(self) -> Self:
        var result = self.copy()
        result.invert()

        return result^

    fn spectrum(self) -> Matrix[DType.float64, color_space.channels(), complex=True]:
        return self._matrix.fourier_transform()

    #
    # Geometric Transformations
    #
    fn flip_horizontally(mut self):
        self._matrix.flip_horizontally()

    fn flipped_horizontally(self) -> Self:
        return Self(self._matrix.flipped_horizontally())

    fn flip_vertically(mut self):
        self._matrix.flip_vertically()

    fn flipped_vertically(self) -> Self:
        return Self(self._matrix.flipped_vertically())

    fn rotate_90[*, clockwise: Bool](mut self):
        self._matrix.rotate_90[clockwise=clockwise]()

    fn rotated_90[*, clockwise: Bool](self) -> Self:
        return Self(self._matrix.rotated_90[clockwise=clockwise]())

    fn rotate_180(mut self):
        self._matrix.rotate_180()

    fn rotated_180(self) -> Self:
        return Self(self._matrix.rotated_180())

    fn scaled[interpolation: Interpolation = Interpolation.bilinear](self, factor: Int) -> Self:
        return self.resized[interpolation](width=factor * self.width(), height=factor * self.height())

    fn scaled[T: Floatable, //, interpolation: Interpolation = Interpolation.bilinear](self, factor: T) -> Self:
        return self.resized[interpolation](width=Int(factor.__float__() * self.width()), height=Int(factor.__float__() * self.height()))

    fn resized[interpolation: Interpolation = Interpolation.bilinear](self, width: Int, height: Int) -> Self:
        @parameter
        if interpolation == Interpolation.nearest:
            return self._resized_nearest(width=width, height=height)
        elif interpolation == Interpolation.bilinear:
            return self._resized_bilinear(width=width, height=height)
        else:
            fatal_error("Unimplemented interpolation for Image.resized(): ", interpolation)
            while True:
                pass

    fn _resized_nearest(self, width: Int, height: Int) -> Self:
        var result = Self(width=width, height=height)

        @parameter
        for channel in range(color_space.channels()):

            @parameter
            fn process_row(y: Int):
                @parameter
                fn process_col[simd_width: Int](x: Int):
                    try:
                        result.strided_store(
                            self._matrix.strided_gather(
                                row=y * self.height() // height,
                                col=x * self.width() // width,
                                component=channel,
                                offset=SIMDRange[simd_width]() * self.width() // width,
                                mask=True,
                            ).value,
                            y=y,
                            x=x,
                            channel=channel,
                        )
                    except error:
                        fatal_error(error)

                vectorize[process_col, Self.optimal_simd_width, unroll_factor=unroll_factor](result.width())

            parallelize[process_row](result.height())

        return result^

    fn _resized_bilinear(self, width: Int, height: Int) -> Self:
        var result = Self(width=width, height=height)

        @parameter
        for channel in range(color_space.channels()):

            @parameter
            fn process_row(y: Int):
                var fractional_y = y * self.height() / height
                var y1 = Int(floor(fractional_y))
                var y2 = min(y1 + 1, self.height() - 1)

                @parameter
                fn process_col[simd_width: Int](x: Int):
                    try:
                        var fractional_x = (x + SIMDRange[simd_width]().cast[DType.float64]()) * self.width() / width
                        var x1 = floor(fractional_x).cast[DType.index]()
                        var x2 = (x1 + 1).clamp(0, self.width() - 1)
                        var x_in_bounds = (x1 != x2)

                        var top_left = self._matrix.strided_gather(row=y1, col=Int(x1[0]), component=channel, offset=x1 - x1[0], mask=True).value.cast[
                            DType.float64
                        ]()
                        var top_right = self._matrix.strided_gather(row=y1, col=Int(x2[0]), component=channel, offset=x2 - x2[0], mask=True).value.cast[
                            DType.float64
                        ]()

                        var top_intermediate = x_in_bounds.select(
                            true_case=(x2.cast[DType.float64]() - fractional_x) * top_left + (fractional_x - x1.cast[DType.float64]()) * top_right,
                            false_case=top_left,
                        )

                        var value: SIMD[DType.float64, simd_width]
                        if y1 == y2:
                            value = top_intermediate
                        else:
                            var bottom_left = self._matrix.strided_gather(row=y2, col=Int(x1[0]), component=channel, offset=x1 - x1[0], mask=True).value.cast[
                                DType.float64
                            ]()
                            var bottom_right = self._matrix.strided_gather(row=y2, col=Int(x2[0]), component=channel, offset=x2 - x2[0], mask=True).value.cast[
                                DType.float64
                            ]()

                            var bottom_intermediate = x_in_bounds.select(
                                true_case=(x2.cast[DType.float64]() - fractional_x) * bottom_left + (fractional_x - x1.cast[DType.float64]()) * bottom_right,
                                false_case=bottom_left,
                            )

                            value = (y2 - fractional_y) * top_intermediate + (fractional_y - y1) * bottom_intermediate

                        result.strided_store(value.cast[dtype](), y=y, x=x, channel=channel)

                    except error:
                        fatal_error(error)

                vectorize[process_col, Self.optimal_simd_width, unroll_factor=unroll_factor](result.width())

            parallelize[process_row](result.height())

        return result^

    fn padded[border: Border = Border.zero](self, size: Int) -> Self:
        return self.padded[border](width=size, height=size)

    fn padded[border: Border = Border.zero](self, width: Int, height: Int) -> Self:
        var result = Self(self._matrix.padded(rows=height, cols=width))

        @parameter
        if border != Border.zero:
            try:

                @parameter
                for channel in range(color_space.channels()):
                    for y in range(height):
                        for x in range(result.width()):
                            result[y, x, channel] = self._bordered_load[border](y=y - height, x=x - width, channel=channel)

                    for y in range(height, height + self.height()):
                        for x in range(result.width()):
                            result[y, x, channel] = self._bordered_load[border](y=y - height, x=x - width, channel=channel)

                        for x in range(width + self.width(), result.width()):
                            result[y, x, channel] = self._bordered_load[border](y=y - height, x=x - width, channel=channel)

                    for y in range(height + self.height(), result.height()):
                        for x in range(result.width()):
                            result[y, x, channel] = self._bordered_load[border](y=y - height, x=x - width, channel=channel)
            except error:
                fatal_error(error)

        return result^

    fn padded_trailing[border: Border = Border.zero](self, width: Int, height: Int) -> Self:
        var result = Self(self._matrix.padded_trailing(rows=height, cols=width))

        @parameter
        if border != Border.zero:
            try:

                @parameter
                for channel in range(color_space.channels()):
                    for y in range(self.height()):
                        for x in range(self.width(), result.width()):
                            result[y, x, channel] = self._bordered_load[border](y=y, x=x, channel=channel)

                    for y in range(self.height(), result.height()):
                        for x in range(result.width()):
                            result[y, x, channel] = self._bordered_load[border](y=y, x=x, channel=channel)

            except error:
                fatal_error(error)

        return result^

    fn horizontally_stacked(self, other: Self) -> Self:
        return Self(self._matrix.horizontally_stacked(other._matrix))

    fn vertically_stacked(self, other: Self) -> Self:
        return Self(self._matrix.vertically_stacked(other._matrix))

    #
    # Common Filters
    #
    fn box_blur[border: Border](mut self, size: Int):
        self.filter[border](Filters.box_kernel_2d[dtype, color_space.channels()](size))

    fn box_blurred[border: Border](self, size: Int) -> Self:
        return self.filtered[border](Filters.box_kernel_2d[dtype, color_space.channels()](size))

    fn gaussian_blur[border: Border](mut self, size: Int, std_dev: Optional[Float64] = None):
        self.filter[border](Filters.gaussian_kernel_2d[dtype, color_space.channels()](size=size, std_dev=std_dev))

    fn gaussian_blurred[border: Border](self, size: Int, std_dev: Optional[Float64] = None) -> Self:
        return self.filtered[border](Filters.gaussian_kernel_2d[dtype, color_space.channels()](size=size, std_dev=std_dev))

    #
    # Filtering
    #
    fn filter[border: Border](mut self, kernel: Matrix[dtype, color_space.channels()]):
        self.filtered[border](kernel).copy_into(self)

    fn filtered[border: Border](self, kernel: Matrix[dtype, color_space.channels()]) -> Self:
        var result = Self(width=self.width(), height=self.height())
        var count = kernel.strided_count()

        if count == 1:
            self._direct_convolution[border, 1](dest=result, kernel=kernel.rotated_180())
        elif count == 2:
            self._direct_convolution[border, 2](dest=result, kernel=kernel.rotated_180())
        elif count <= 4:
            self._direct_convolution[border, 4](dest=result, kernel=kernel.rotated_180())
        elif count <= 8:
            self._direct_convolution[border, 8](dest=result, kernel=kernel.rotated_180())
        elif count <= 16:
            self._direct_convolution[border, 16](dest=result, kernel=kernel.rotated_180())
        elif count <= 32:
            self._direct_convolution[border, 32](dest=result, kernel=kernel.rotated_180())
        elif count <= 64:
            self._direct_convolution[border, 64](dest=result, kernel=kernel.rotated_180())
        elif count <= 128:
            self._direct_convolution[border, 128](dest=result, kernel=kernel.rotated_180())
        else:
            fatal_error("Direct convolution for kernels with strided counts greater than 128 is not supported yet")

        return result^

    fn _direct_convolution[border: Border, width: Int](self, mut dest: Self, kernel: Matrix[dtype, color_space.channels()]):
        var half_kernel_width = kernel.cols() // 2
        var half_kernel_height = kernel.rows() // 2

        var kernel_vector = SIMD[dtype, width]()
        var y_offset = SIMD[DType.index, width]()
        var x_offset = SIMD[DType.index, width]()
        var offset = SIMD[DType.index, width]()
        var mask = SIMD[DType.bool, width](False)

        for kernel_y in range(0, kernel.rows()):
            for kernel_x in range(0, kernel.cols()):
                var index = kernel_y * kernel.cols() + kernel_x
                y_offset[index] = kernel_y - half_kernel_height
                x_offset[index] = kernel_x - half_kernel_width
                offset[index] = (y_offset[index] * self.width() + x_offset[index]) * color_space.channels()
                mask[index] = True

        @parameter
        for channel in range(color_space.channels()):
            # TODO: Create strided slice of kernel data and fill the kernel vector from that, or create an as_simd() method on Matrix?
            for row in range(kernel.rows()):
                for col in range(kernel.cols()):
                    try:
                        kernel_vector[row * kernel.cols() + col] = kernel.strided_load[1](row=row, col=col, component=channel).value
                    except error:
                        fatal_error(error)

            @parameter
            fn process_row(y: Int):
                var min_patch_y = y - half_kernel_height
                var max_patch_y = y + half_kernel_height

                try:
                    for x in range(dest.width()):
                        if max_patch_y < self.height() and min_patch_y >= 0 and (x + half_kernel_width) < self.width() and (x - half_kernel_width) >= 0:
                            dest[y, x, channel] = (
                                self._matrix.gather(
                                    row=y,
                                    col=x,
                                    component=channel,
                                    offset=offset,
                                    mask=mask,
                                )
                                * kernel_vector
                            ).value.reduce_add()
                        else:
                            var pixel_sum = Scalar[dtype]()

                            @parameter
                            for index in range(width):
                                if mask[index]:
                                    pixel_sum = kernel_vector[index].fma(
                                        self._bordered_load[border](
                                            y=Int(y + y_offset[index]),
                                            x=Int(x + x_offset[index]),
                                            channel=channel,
                                        ),
                                        pixel_sum,
                                    )

                            dest[y, x, channel] = pixel_sum
                except error:
                    fatal_error(error)

            parallelize[process_row](dest.height())

    fn _discrete_fourier_transform_convolution[
        border: Border, new_dtype: DType
    ](self, mut dest: Image[new_dtype, color_space], flipped_kernel: Matrix[dtype, color_space.channels()],):
        pass

    @always_inline
    fn _bordered_load[border: Border](self, y: Int, x: Int, channel: Int) -> Scalar[dtype]:
        try:

            @parameter
            if border == Border.zero:
                if y >= 0 and y < self.height() and x >= 0 and x < self.width():
                    return self[y, x, channel]
                else:
                    return Scalar[dtype](0)
            elif border == Border.wrap:
                return self[y % self.height(), x % self.width(), channel]
            elif border == Border.reflect:
                var reflected_y: Int
                var reflected_x: Int

                if y < 0:
                    reflected_y = abs(y)
                elif y >= self.height():
                    reflected_y = 2 * (self.height() - 1) - y
                else:
                    reflected_y = y

                if x < 0:
                    reflected_x = abs(x)
                elif x >= self.width():
                    reflected_x = 2 * (self.width() - 1) - x
                else:
                    reflected_x = x

                return self[reflected_y, reflected_x, channel]
            else:
                fatal_error("Unimplemented border type in Image._bordered_load()")
                while True:
                    pass
        except error:
            fatal_error(error)
            while True:
                pass

    #
    # Type Conversion
    #
    fn astype[new_dtype: DType](self) -> Image[new_dtype, color_space]:
        return Image[new_dtype, color_space](self._matrix.astype[new_dtype]())

    #
    # Color Space Conversion
    #
    fn converted[new_color_space: ColorSpace](self) -> Image[dtype, new_color_space]:
        return self.converted_astype[dtype, new_color_space]()

    fn converted_astype[new_dtype: DType, new_color_space: ColorSpace](self) -> Image[new_dtype, new_color_space]:
        @parameter
        if new_color_space == color_space:
            return Image[new_dtype, new_color_space](self._matrix.rebound_copy[new_dtype, new_color_space.channels()]())
        else:
            var result = Image[new_dtype, new_color_space](width=self.width(), height=self.height())

            @parameter
            fn convert_row(y: Int):
                @parameter
                fn convert_cols[width: Int](x: Int):
                    try:
                        # Greyscale ->
                        @parameter
                        if color_space == ColorSpace.greyscale:
                            var grey = self.strided_load[width](y=y, x=x, channel=0).cast[new_dtype]()

                            # RGB
                            @parameter
                            if new_color_space == ColorSpace.rgb:
                                result.strided_store(grey, y=y, x=x, channel=0)
                                result.strided_store(grey, y=y, x=x, channel=1)
                                result.strided_store(grey, y=y, x=x, channel=2)

                        # RGB ->
                        elif color_space == ColorSpace.rgb:
                            var red = self.strided_load[width](y=y, x=x, channel=0).cast[DType.float64]()
                            var green = self.strided_load[width](y=y, x=x, channel=1).cast[DType.float64]()
                            var blue = self.strided_load[width](y=y, x=x, channel=2).cast[DType.float64]()

                            # Greyscale
                            @parameter
                            if new_color_space == ColorSpace.greyscale:
                                var grey = 0.299 * red + 0.587 * green + 0.114 * blue
                                result.strided_store(grey.cast[new_dtype](), y=y, x=x, channel=0)

                    except error:
                        fatal_error(error)

                vectorize[convert_cols, Self.optimal_simd_width, unroll_factor=unroll_factor](self.width())

            parallelize[convert_row](num_work_items=self.height())

            return result^

    #
    # Saving to File
    #
    fn save[file_type: ImageFile](self, path: StringLiteral) raises:
        self.save[file_type](Path(path))

    fn save[file_type: ImageFile](self, path: Path) raises:
        ImageWriter(path).write[file_type](self)

    #
    # Stringable & Writable
    #
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[Image: width = ", self.width(), ", height = ", self.height(), ", color_space = ", color_space, ", bit_depth = ", dtype.bitwidth(), "]")
