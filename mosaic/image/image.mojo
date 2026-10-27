#
# image.mojo
# mosaic
#
# Created by Christian Bator on 12/14/2024
#

from std.os import abort
from std.pathlib import Path
from std.algorithm import vectorize
from max.algorithm import parallelize
from std.math import floor, ceil, ceildiv, trunc, Ceilable, CeilDivable, Floorable, Truncable
from std.sys.info import bit_width_of

from mosaic.numeric import Matrix, MatrixSlice, StridedRange, SIMDRange, Number, ScalarNumber
from mosaic.numeric.fft import fft_dtype
from mosaic.utility import optimal_simd_width, unroll_factor, _assert

from .image_reader import ImageReader
from .image_writer import ImageWriter
from .filters import Filters
from .codec import ImageFile
from .color_space import ColorSpace
from .border import Border
from .image_slice import ImageSlice
from .interpolation import Interpolation


#
# Image
#
struct Image[color_space: ColorSpace, dtype: DType](
    Absable, CeilDivable, Ceilable, Copyable, Equatable, Floorable, Movable, Roundable, Truncable, Writable
):
    #
    # Fields
    #
    comptime optimal_simd_width = optimal_simd_width[Self.dtype]()

    var _matrix: Matrix[Self.dtype, Self.color_space.channels()]

    #
    # Initialization
    #
    def __init__(out self, path: StringLiteral) raises:
        self = Self(Path(path))

    def __init__(out self, path: Path) raises:
        self = ImageReader[Self.color_space, Self.dtype](path).read()

    def __init__(out self, height: Int, width: Int):
        self._matrix = Matrix[Self.dtype, Self.color_space.channels()](rows=height, cols=width)

    # This is an unsafe convenience constructor
    def __init__(out self, height: Int, width: Int, var data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]):
        self._matrix = Matrix[Self.dtype, Self.color_space.channels()](rows=height, cols=width, data=data)

    def __init__(out self, var matrix: Matrix[Self.dtype, Self.color_space.channels()]):
        self._matrix = matrix^

    def __init__(out self, *, deinit move: Self):
        self._matrix = move._matrix^

    @staticmethod
    def with_single_channel_data[channel: Int](single_channel_matrix: Matrix[Self.dtype]) -> Self:
        return Self(single_channel_matrix.copied_to_component[channel, Self.color_space.channels()]())

    @staticmethod
    def from_spectrum[
        spectrum_dtype: DType, //
    ](
        spectrum: Matrix[spectrum_dtype, Self.color_space.channels(), complex=True],
        lower_bound: Scalar[DType.int64],
        upper_bound: Scalar[DType.int64],
    ) -> Self:
        var real = spectrum.fourier_transform[inverse=True]().real()
        real.map_to_range(lower_bound.cast[fft_dtype](), upper_bound.cast[fft_dtype]())

        return Self(real.as_type[Self.dtype]())

    @staticmethod
    def from_spectrum[
        spectrum_dtype: DType, T: Floatable, //
    ](spectrum: Matrix[spectrum_dtype, Self.color_space.channels(), complex=True], lower_bound: T, upper_bound: T) -> Self:
        var real = spectrum.fourier_transform[inverse=True]().real()
        real.map_to_range(Float64(lower_bound).cast[fft_dtype](), Float64(upper_bound).cast[fft_dtype]())

        return Self(real.as_type[Self.dtype]())

    #
    # Properties
    #
    @always_inline
    def height(self) -> Int:
        return self._matrix.rows()

    @always_inline
    def width(self) -> Int:
        return self._matrix.cols()

    @always_inline
    def pixels(self) -> Int:
        return self._matrix.strided_count()

    def channels(self) -> Int:
        return Self.color_space.channels()

    @always_inline
    def samples(self) -> Int:
        return self._matrix.count()

    def bit_depth(self) -> Int:
        return bit_width_of[Self.dtype]()

    @always_inline
    def bytes(self) -> Int:
        return self.samples() * self.bit_depth() // 8

    #
    # Public Access
    #
    @always_inline
    def __getitem__(self, y: Int, x: Int) raises -> Scalar[Self.dtype]:
        comptime assert Self.color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"

        return self[y, x, 0]

    @always_inline
    def __getitem__(self, y: Int, x: Int, channel: Int) raises -> Scalar[Self.dtype]:
        return self._matrix[y, x, channel].value

    @always_inline
    def __setitem__(mut self, y: Int, x: Int, value: Scalar[Self.dtype]) raises:
        comptime assert Self.color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"

        self[y, x, 0] = value

    @always_inline
    def __setitem__(mut self, y: Int, x: Int, channel: Int, value: Scalar[Self.dtype]) raises:
        self._matrix[y, x, channel] = value

    @always_inline
    def strided_load[width: Int = 1](self, y: Int, x: Int) raises -> SIMD[Self.dtype, width]:
        comptime assert Self.color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"

        return self.strided_load[width](y=y, x=x, channel=0)

    @always_inline
    def strided_load[width: Int = 1](self, y: Int, x: Int, channel: Int) raises -> SIMD[Self.dtype, width]:
        return self._matrix.strided_load[width](row=y, col=x, component=channel).value

    @always_inline
    def strided_store[width: SIMDLength](mut self, value: SIMD[Self.dtype, width], y: Int, x: Int) raises:
        comptime assert Self.color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"

        self.strided_store(value, y=y, x=x, channel=0)

    @always_inline
    def strided_store[width: SIMDLength](mut self, value: SIMD[Self.dtype, width], y: Int, x: Int, channel: Int) raises:
        self._matrix.strided_store(Number[Self.dtype, width](value), row=y, col=x, component=channel)

    #
    # Private Access
    #
    @always_inline
    def _strided_load[width: Int = 1](self, y: Int, x: Int) -> SIMD[Self.dtype, width]:
        comptime assert Self.color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"

        return self._strided_load[width](y=y, x=x, channel=0)

    @always_inline
    def _strided_load[width: Int = 1](self, y: Int, x: Int, channel: Int) -> SIMD[Self.dtype, width]:
        return self._matrix._strided_load[width](row=y, col=x, component=channel).value

    @always_inline
    def _strided_store[width: SIMDLength](mut self, value: SIMD[Self.dtype, width], y: Int, x: Int):
        comptime assert Self.color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"

        self._strided_store(value, y=y, x=x, channel=0)

    @always_inline
    def _strided_store[width: SIMDLength](mut self, value: SIMD[Self.dtype, width], y: Int, x: Int, channel: Int):
        self._matrix._strided_store(Number[Self.dtype, width](value), row=y, col=x, component=channel)

    #
    # Unsafe Access
    #
    @always_inline
    def unsafe_data_ptr(self) -> Pointer[Scalar[Self.dtype], MutUntrackedOrigin]:
        return self._matrix.unsafe_data_ptr()

    @always_inline
    def unsafe_uint8_ptr(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        return self._matrix.unsafe_uint8_ptr()

    def keep(self):
        pass

    #
    # Slicing
    #
    @always_inline
    def __getitem__[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self, y: Int, x_slice: Slice) raises -> ImageSlice[Self.color_space, Self.dtype, origin]:
        return self[y : y + 1, x_slice]

    @always_inline
    def __getitem__[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self, y_slice: Slice, x: Int) raises -> ImageSlice[Self.color_space, Self.dtype, origin]:
        return self[y_slice, x : x + 1]

    @always_inline
    def __getitem__[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self, y_slice: Slice, x_slice: Slice) raises -> ImageSlice[Self.color_space, Self.dtype, origin]:
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
    def slice[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self, y_range: StridedRange) raises -> ImageSlice[Self.color_space, Self.dtype, origin]:
        return self.slice(y_range=y_range, x_range=StridedRange(self.width()))

    @always_inline
    def slice[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self, *, x_range: StridedRange) raises -> ImageSlice[Self.color_space, Self.dtype, origin]:
        return self.slice(y_range=StridedRange(self.height()), x_range=x_range)

    @always_inline
    def slice[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self, y_range: StridedRange, x_range: StridedRange) raises -> ImageSlice[Self.color_space, Self.dtype, origin]:
        return ImageSlice[Self.color_space, Self.dtype, origin](self, y_range=y_range, x_range=x_range)

    @always_inline
    def channel_slice[
        channel: Int
    ](self) -> MatrixSlice[StridedRange(channel, channel + 1), Self.dtype, Self.color_space.channels(), False, origin_of(self._matrix)]:
        return self._matrix.component_slice[channel]()

    @always_inline
    def channel_slice[
        channel: Int
    ](self, y_range: StridedRange) raises -> MatrixSlice[
        StridedRange(channel, channel + 1), Self.dtype, Self.color_space.channels(), False, origin_of(self._matrix)
    ]:
        return self._matrix.component_slice[channel](y_range)

    @always_inline
    def channel_slice[
        channel: Int
    ](self, *, x_range: StridedRange) raises -> MatrixSlice[
        StridedRange(channel, channel + 1), Self.dtype, Self.color_space.channels(), False, origin_of(self._matrix)
    ]:
        return self._matrix.component_slice[channel](col_range=x_range)

    @always_inline
    def channel_slice[
        channel: Int
    ](self, y_range: StridedRange, x_range: StridedRange) raises -> MatrixSlice[
        StridedRange(channel, channel + 1), Self.dtype, Self.color_space.channels(), False, origin_of(self._matrix)
    ]:
        return self._matrix.component_slice[channel](row_range=y_range, col_range=x_range)

    @always_inline
    def strided_slice[
        channel_range: StridedRange
    ](self) -> MatrixSlice[channel_range, Self.dtype, Self.color_space.channels(), False, origin_of(self._matrix)]:
        return self._matrix.strided_slice[channel_range]()

    @always_inline
    def strided_slice[
        channel_range: StridedRange
    ](self, y_range: StridedRange) raises -> MatrixSlice[
        channel_range, Self.dtype, Self.color_space.channels(), False, origin_of(self._matrix)
    ]:
        return self._matrix.strided_slice[channel_range](y_range)

    @always_inline
    def strided_slice[
        channel_range: StridedRange
    ](self, *, x_range: StridedRange) raises -> MatrixSlice[
        channel_range, Self.dtype, Self.color_space.channels(), False, origin_of(self._matrix)
    ]:
        return self._matrix.strided_slice[channel_range](col_range=x_range)

    @always_inline
    def strided_slice[
        channel_range: StridedRange
    ](self, y_range: StridedRange, x_range: StridedRange) raises -> MatrixSlice[
        channel_range, Self.dtype, Self.color_space.channels(), False, origin_of(self._matrix)
    ]:
        return self._matrix.strided_slice[channel_range](row_range=y_range, col_range=x_range)

    @always_inline
    def extract_channel[channel: Int](self) -> Matrix[Self.dtype]:
        return self.channel_slice[channel]().deep_copy[rebound_depth=1]()

    def extract_channel(self, channel: Int) -> Matrix[Self.dtype]:
        _assert(0 <= channel < Self.color_space.channels(), "Channel must be in color space channel bounds")

        if channel == 0:
            return self.extract_channel[0]()
        elif channel == 1:
            return self.extract_channel[1]()
        elif channel == 2:
            return self.extract_channel[2]()
        else:
            abort("Unimplemented runtime extract_channel")

    def store_sub_image(mut self, value: Self, y: Int, x: Int) raises:
        self.store_sub_image(value[:, :], y=y, x=x)

    def store_sub_image(mut self, image_slice: ImageSlice[Self.color_space, Self.dtype, _], y: Int, x: Int) raises:
        if (image_slice.y_range().end > self.height()) or (image_slice.x_range().end > self.width()):
            raise Error("Out of bounds sub-image store")

        comptime for channel in range(Self.color_space.channels()):

            def store_row(image_slice_y: Int) {mut self, imm}:
                def store_cols[width: Int](image_slice_x: Int) {mut self, imm image_slice, imm image_slice_y, imm y, imm x}:
                    self._strided_store(
                        image_slice._strided_load[width](y=image_slice_y, x=image_slice_x, channel=channel),
                        y=y + image_slice_y,
                        x=x + image_slice_x,
                        channel=channel,
                    )

                vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](image_slice.width(), store_cols)

            parallelize(store_row, image_slice.height())

    def store_sub_matrix(mut self, value: Matrix[Self.dtype, Self.color_space.channels()], y: Int, x: Int, channel: Int = 0) raises:
        self._matrix.store_sub_matrix(value, row=y, col=x, component=channel)

    def store_sub_matrix(
        mut self,
        value: MatrixSlice[_, Self.dtype, Self.color_space.channels(), False, _],
        y: Int,
        x: Int,
        channel: Int = 0,
    ) raises:
        self._matrix.store_sub_matrix(value, row=y, col=x, component=channel)

    #
    # Copyable
    #
    def __init__(out self, *, copy: Self):
        self = Self(copy._matrix.copy())

    def copy_into(self, mut other: Self):
        self._matrix.copy_into(other._matrix)

    #
    # Equatable
    #
    def __eq__(self, other: Self) -> Bool:
        return self._matrix == other._matrix

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def matches(self, other: Self, tolerance: Int = 1) -> Bool:
        return self._matrix.matches(other._matrix, tolerance)

    #
    # Operators (Scalar)
    #
    def __neg__(self) -> Self:
        return self * -1

    def __add__(self, rhs: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix + rhs)

    def __iadd__(mut self, rhs: Scalar[Self.dtype]):
        self._matrix += rhs

    def __sub__(self, rhs: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix - rhs)

    def __isub__(mut self, rhs: Scalar[Self.dtype]):
        self._matrix -= rhs

    def __mul__(self, rhs: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix * rhs)

    def __rmul__(self, lhs: Scalar[Self.dtype]) -> Self:
        return self * lhs

    def __imul__(mut self, rhs: Scalar[Self.dtype]):
        self._matrix *= rhs

    def __truediv__(self, rhs: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix / rhs)

    def __itruediv__(mut self, rhs: Scalar[Self.dtype]):
        self._matrix /= rhs

    def __floordiv__(self, rhs: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix // rhs)

    def __ifloordiv__(mut self, rhs: Scalar[Self.dtype]):
        self._matrix //= rhs

    def __mod__(self, rhs: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix % rhs)

    def __imod__(mut self, rhs: Scalar[Self.dtype]):
        self._matrix %= rhs

    def __pow__(self, rhs: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix**rhs)

    def __ipow__(mut self, rhs: Scalar[Self.dtype]):
        self._matrix **= rhs

    def __and__(self, rhs: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix & rhs)

    def __iand__(mut self, rhs: Scalar[Self.dtype]):
        self._matrix &= rhs

    def __or__(self, rhs: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix | rhs)

    def __ior__(mut self, rhs: Scalar[Self.dtype]):
        self._matrix |= rhs

    def __xor__(self, rhs: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix ^ rhs)

    def __ixor__(mut self, rhs: Scalar[Self.dtype]):
        self._matrix ^= rhs

    def __lshift__(self, rhs: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix << rhs)

    def __ilshift__(mut self, rhs: Scalar[Self.dtype]):
        self._matrix <<= rhs

    def __rshift__(self, rhs: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix >> rhs)

    def __irshift__(mut self, rhs: Scalar[Self.dtype]):
        self._matrix >>= rhs

    #
    # Operators (Matrix)
    #
    def __add__(self, rhs: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        return Self(self._matrix + rhs)

    def __iadd__(mut self, rhs: Matrix[Self.dtype, Self.color_space.channels()]):
        self._matrix += rhs

    def __sub__(self, rhs: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        return Self(self._matrix - rhs)

    def __isub__(mut self, rhs: Matrix[Self.dtype, Self.color_space.channels()]):
        self._matrix -= rhs

    def __truediv__(self, rhs: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        return Self(self._matrix / rhs)

    def __itruediv__(mut self, rhs: Matrix[Self.dtype, Self.color_space.channels()]):
        self._matrix /= rhs

    def __floordiv__(self, rhs: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        return Self(self._matrix // rhs)

    def __ifloordiv__(mut self, rhs: Matrix[Self.dtype, Self.color_space.channels()]):
        self._matrix //= rhs

    def __mod__(self, rhs: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        return Self(self._matrix % rhs)

    def __imod__(mut self, rhs: Matrix[Self.dtype, Self.color_space.channels()]):
        self._matrix %= rhs

    def __pow__(self, rhs: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        return Self(self._matrix**rhs)

    def __ipow__(mut self, rhs: Matrix[Self.dtype, Self.color_space.channels()]):
        self._matrix **= rhs

    def __and__(self, rhs: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        return Self(self._matrix & rhs)

    def __iand__(mut self, rhs: Matrix[Self.dtype, Self.color_space.channels()]):
        self._matrix &= rhs

    def __or__(self, rhs: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        return Self(self._matrix | rhs)

    def __ior__(mut self, rhs: Matrix[Self.dtype, Self.color_space.channels()]):
        self._matrix |= rhs

    def __xor__(self, rhs: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        return Self(self._matrix ^ rhs)

    def __ixor__(mut self, rhs: Matrix[Self.dtype, Self.color_space.channels()]):
        self._matrix ^= rhs

    def __lshift__(self, rhs: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        return Self(self._matrix << rhs)

    def __ilshift__(mut self, rhs: Matrix[Self.dtype, Self.color_space.channels()]):
        self._matrix <<= rhs

    def __rshift__(self, rhs: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        return Self(self._matrix >> rhs)

    def __irshift__(mut self, rhs: Matrix[Self.dtype, Self.color_space.channels()]):
        self._matrix >>= rhs

    #
    # Operators (Image)
    #
    def __add__(self, rhs: Self) -> Self:
        return Self(self._matrix + rhs._matrix)

    def __iadd__(mut self, rhs: Self):
        self._matrix += rhs._matrix

    def __sub__(self, rhs: Self) -> Self:
        return Self(self._matrix - rhs._matrix)

    def __isub__(mut self, rhs: Self):
        self._matrix -= rhs._matrix

    def __truediv__(self, rhs: Self) -> Self:
        return Self(self._matrix / rhs._matrix)

    def __itruediv__(mut self, rhs: Self):
        self._matrix /= rhs._matrix

    def __floordiv__(self, rhs: Self) -> Self:
        return Self(self._matrix // rhs._matrix)

    def __ifloordiv__(mut self, rhs: Self):
        self._matrix //= rhs._matrix

    def __mod__(self, rhs: Self) -> Self:
        return Self(self._matrix % rhs._matrix)

    def __imod__(mut self, rhs: Self):
        self._matrix %= rhs._matrix

    def __pow__(self, rhs: Self) -> Self:
        return Self(self._matrix**rhs._matrix)

    def __ipow__(mut self, rhs: Self):
        self._matrix **= rhs._matrix

    def __and__(self, rhs: Self) -> Self:
        return Self(self._matrix & rhs._matrix)

    def __iand__(mut self, rhs: Self):
        self._matrix &= rhs._matrix

    def __or__(self, rhs: Self) -> Self:
        return Self(self._matrix | rhs._matrix)

    def __ior__(mut self, rhs: Self):
        self._matrix |= rhs._matrix

    def __xor__(self, rhs: Self) -> Self:
        return Self(self._matrix ^ rhs._matrix)

    def __ixor__(mut self, rhs: Self):
        self._matrix ^= rhs._matrix

    def __lshift__(self, rhs: Self) -> Self:
        return Self(self._matrix << rhs._matrix)

    def __ilshift__(mut self, rhs: Self):
        self._matrix <<= rhs._matrix

    def __rshift__(self, rhs: Self) -> Self:
        return Self(self._matrix >> rhs._matrix)

    def __irshift__(mut self, rhs: Self):
        self._matrix >>= rhs._matrix

    #
    # Numeric Traits
    #
    def __floor__(self) -> Self:
        return Self(floor(self._matrix))

    def __ceil__(self) -> Self:
        return Self(ceil(self._matrix))

    def __ceildiv__(self, rhs: Self) -> Self:
        return Self(ceildiv(self._matrix, rhs._matrix))

    def __round__(self) -> Self:
        return Self(round(self._matrix))

    def __round__(self, ndigits: Int) -> Self:
        return Self(round(self._matrix, ndigits=ndigits))

    def __trunc__(self) -> Self:
        return Self(trunc(self._matrix))

    def __abs__(self) -> Self:
        return Self(abs(self._matrix))

    #
    # Numeric Methods
    #
    def log(self) -> Self where Self.dtype.is_floating_point():
        return Self(self._matrix.log())

    def clamp(mut self, lower_bound: Scalar[Self.dtype], upper_bound: Scalar[Self.dtype]):
        self._matrix.clamp(lower_bound=lower_bound, upper_bound=upper_bound)

    def clamped(self, lower_bound: Scalar[Self.dtype], upper_bound: Scalar[Self.dtype]) -> Self:
        return Self(self._matrix.clamped(lower_bound=lower_bound, upper_bound=upper_bound))

    def strided_sum(self, channel: Int) -> Scalar[Self.dtype]:
        return self._matrix.strided_sum(channel).value

    def strided_average(self, channel: Int) -> Scalar[DType.float64]:
        return self._matrix.strided_average(channel).value

    def strided_min(self, channel: Int) -> Scalar[Self.dtype]:
        return self._matrix.strided_min(channel).value

    def strided_max(self, channel: Int) -> Scalar[Self.dtype]:
        return self._matrix.strided_max(channel).value

    def strided_normalize(mut self):
        self._matrix.strided_normalize()

    def strided_fill(mut self, value: Scalar[Self.dtype], channel: Int):
        self._matrix.strided_fill(value, component=channel)

    def fill(mut self, value: Scalar[Self.dtype]):
        self._matrix.fill(value)

    def strided_for_each[
        TransformerType: def[width: Int](value: SIMD[Self.dtype, width]) -> SIMD[Self.dtype, width], //
    ](mut self, transformer: TransformerType, channel: Int):
        def number_transformer[width: Int](value: Number[Self.dtype, width]) {imm} -> Number[Self.dtype, width]:
            return Number[Self.dtype, width](transformer[width](value.value))

        self._matrix.strided_for_each(number_transformer, channel)

    def strided_for_each_zipped[
        TransformerType: def[width: Int](value: SIMD[Self.dtype, width], rhs: SIMD[Self.dtype, width]) -> SIMD[Self.dtype, width], //
    ](mut self, transformer: TransformerType, other: Self, channel: Int):
        def number_transformer[
            width: Int
        ](value: Number[Self.dtype, width], rhs: Number[Self.dtype, width]) {imm} -> Number[Self.dtype, width]:
            return Number[Self.dtype, width](transformer[width](value=value.value, rhs=rhs.value))

        self._matrix.strided_for_each_zipped(number_transformer, other._matrix, component=channel)

    def for_each[
        TransformerType: def[width: Int](value: SIMD[Self.dtype, width]) -> SIMD[Self.dtype, width], //
    ](mut self, transformer: TransformerType):
        def number_transformer[width: Int](value: Number[Self.dtype, width]) {imm} -> Number[Self.dtype, width]:
            return Number[Self.dtype, width](transformer[width](value.value))

        self._matrix.for_each(number_transformer)

    def for_each_zipped[
        TransformerType: def[width: Int](value: SIMD[Self.dtype, width], rhs: SIMD[Self.dtype, width]) -> SIMD[Self.dtype, width], //
    ](mut self, transformer: TransformerType, other: Self):
        def number_transformer[
            width: Int
        ](value: Number[Self.dtype, width], rhs: Number[Self.dtype, width]) {imm} -> Number[Self.dtype, width]:
            return Number[Self.dtype, width](transformer[width](value=value.value, rhs=rhs.value))

        self._matrix.for_each_zipped(number_transformer, other._matrix)

    def invert(mut self):
        comptime assert Self.dtype == DType.uint8, "invert() is only available for UInt8 images, since inversion is ambiguous otherwise"

        def transformer[width: Int](value: SIMD[Self.dtype, width]) {imm} -> SIMD[Self.dtype, width]:
            return 255 - value

        self.for_each(transformer)

    def inverted(self) -> Self:
        var result = self.copy()
        result.invert()

        return result^

    def spectrum(self) -> Matrix[fft_dtype, Self.color_space.channels(), complex=True]:
        return self._matrix.fourier_transform()

    #
    # Geometric Transformations
    #
    def flip_horizontally(mut self):
        self._matrix.flip_horizontally()

    def flipped_horizontally(self) -> Self:
        return Self(self._matrix.flipped_horizontally())

    def flip_vertically(mut self):
        self._matrix.flip_vertically()

    def flipped_vertically(self) -> Self:
        return Self(self._matrix.flipped_vertically())

    def rotate_90[*, clockwise: Bool](mut self):
        self._matrix.rotate_90[clockwise=clockwise]()

    def rotated_90[*, clockwise: Bool](self) -> Self:
        return Self(self._matrix.rotated_90[clockwise=clockwise]())

    def rotate_180(mut self):
        self._matrix.rotate_180()

    def rotated_180(self) -> Self:
        return Self(self._matrix.rotated_180())

    def scaled[interpolation: Interpolation = Interpolation.bilinear](self, factor: Float64) -> Self:
        return self.resized[interpolation](height=Int(factor * Float64(self.height())), width=Int(factor * Float64(self.width())))

    def scaled[T: Floatable, //, interpolation: Interpolation = Interpolation.bilinear](self, factor: T) -> Self:
        return self.resized[interpolation](
            height=Int(Float64(factor) * Float64(self.height())), width=Int(Float64(factor) * Float64(self.width()))
        )

    def resized[interpolation: Interpolation = Interpolation.bilinear](self, height: Int, width: Int) -> Self:
        comptime if interpolation == Interpolation.nearest:
            return self._resized_nearest(height=height, width=width)
        elif interpolation == Interpolation.bilinear:
            return self._resized_bilinear(height=height, width=width)
        else:
            abort("Unimplemented interpolation for Image.resized(): " + String(interpolation))

    def _resized_nearest(self, height: Int, width: Int) -> Self:
        var result = Self(height=height, width=width)

        comptime for channel in range(Self.color_space.channels()):

            def process_row(y: Int) {mut result, imm}:
                def process_col[simd_width: Int](x: Int) {mut result, imm self, imm height, imm width, imm y}:
                    result._strided_store(
                        self._matrix.strided_gather(
                            row=y * self.height() // height,
                            col=x * self.width() // width,
                            component=channel,
                            offset=SIMDRange[simd_width]() * self.width() // width,
                            mask=SIMD[DType.bool, simd_width](fill=True),
                        ).value,
                        y=y,
                        x=x,
                        channel=channel,
                    )

                vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](result.width(), process_col)

            parallelize(process_row, result.height())

        return result^

    def _resized_bilinear(self, height: Int, width: Int) -> Self:
        var result = Self(height=height, width=width)

        comptime for channel in range(Self.color_space.channels()):

            def process_row(y: Int) {mut result, imm}:
                var fractional_y = Float64(y * self.height()) / Float64(height)
                var y1 = Int(floor(fractional_y))
                var y2 = min(y1 + 1, self.height() - 1)

                def process_col[
                    simd_width: Int
                ](x: Int) {mut result, imm self, imm height, imm width, imm fractional_y, imm y1, imm y2, imm y}:
                    var fractional_x = (Float64(x) + SIMDRange[simd_width]().cast[DType.float64]()) * Float64(self.width()) / Float64(width)
                    var x1 = floor(fractional_x).cast[DType.int]()
                    var x2 = (x1 + 1).clamp(0, self.width() - 1)
                    var x_in_bounds = (x2 - x1).cast[DType.bool]()

                    var top_left = self._matrix.strided_gather(
                        row=y1, col=Int(x1[0]), component=channel, offset=x1 - x1[0], mask=SIMD[DType.bool, simd_width](fill=True)
                    ).value.cast[DType.float64]()
                    var top_right = self._matrix.strided_gather(
                        row=y1, col=Int(x2[0]), component=channel, offset=x2 - x2[0], mask=SIMD[DType.bool, simd_width](fill=True)
                    ).value.cast[DType.float64]()

                    var top_intermediate = x_in_bounds.select(
                        true_case=(x2.cast[DType.float64]() - fractional_x) * top_left
                        + (fractional_x - x1.cast[DType.float64]()) * top_right,
                        false_case=top_left,
                    )

                    var value: SIMD[DType.float64, simd_width]
                    if y1 == y2:
                        value = top_intermediate
                    else:
                        var bottom_left = self._matrix.strided_gather(
                            row=y2, col=Int(x1[0]), component=channel, offset=x1 - x1[0], mask=SIMD[DType.bool, simd_width](fill=True)
                        ).value.cast[DType.float64]()
                        var bottom_right = self._matrix.strided_gather(
                            row=y2, col=Int(x2[0]), component=channel, offset=x2 - x2[0], mask=SIMD[DType.bool, simd_width](fill=True)
                        ).value.cast[DType.float64]()

                        var bottom_intermediate = x_in_bounds.select(
                            true_case=(x2.cast[DType.float64]() - fractional_x) * bottom_left
                            + (fractional_x - x1.cast[DType.float64]()) * bottom_right,
                            false_case=bottom_left,
                        )

                        value = (Float64(y2) - fractional_y) * top_intermediate + (fractional_y - Float64(y1)) * bottom_intermediate

                    result._strided_store(value.cast[Self.dtype](), y=y, x=x, channel=channel)

                vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](result.width(), process_col)

            parallelize(process_row, result.height())

        return result^

    def padded[border: Border = Border.zero](self, size: Int) -> Self:
        return self.padded[border](height=size, width=size)

    def padded[border: Border = Border.zero](self, height: Int, width: Int) -> Self:
        var result = Self(self._matrix.padded(rows=height, cols=width))

        comptime if border != Border.zero:

            def store_value(y: Int, x: Int, channel: Int) {mut result, imm}:
                result._strided_store(self._bordered_load[border](y=y - height, x=x - width, channel=channel), y=y, x=x, channel=channel)

            comptime for channel in range(Self.color_space.channels()):
                for y in range(height):
                    for x in range(result.width()):
                        store_value(y=y, x=x, channel=channel)

                for y in range(height, height + self.height()):
                    for x in range(result.width()):
                        store_value(y=y, x=x, channel=channel)

                    for x in range(width + self.width(), result.width()):
                        store_value(y=y, x=x, channel=channel)

                for y in range(height + self.height(), result.height()):
                    for x in range(result.width()):
                        store_value(y=y, x=x, channel=channel)

        return result^

    def padded_trailing[border: Border = Border.zero](self, height: Int, width: Int) -> Self:
        var result = Self(self._matrix.padded_trailing(rows=height, cols=width))

        comptime if border != Border.zero:
            comptime for channel in range(Self.color_space.channels()):
                for y in range(self.height()):
                    for x in range(self.width(), result.width()):
                        result._strided_store(self._bordered_load[border](y=y, x=x, channel=channel), y=y, x=x, channel=channel)

                for y in range(self.height(), result.height()):
                    for x in range(result.width()):
                        result._strided_store(self._bordered_load[border](y=y, x=x, channel=channel), y=y, x=x, channel=channel)

        return result^

    def horizontally_stacked(self, other: Self) -> Self:
        return Self(self._matrix.horizontally_stacked(other._matrix))

    def vertically_stacked(self, other: Self) -> Self:
        return Self(self._matrix.vertically_stacked(other._matrix))

    #
    # Common Filters
    #
    def box_blur[border: Border](mut self, size: Int) raises:
        self.filter[border](Filters.box_kernel_2d[Self.dtype, Self.color_space.channels()](size))

    def box_blurred[border: Border](self, size: Int) raises -> Self:
        return self.filtered[border](Filters.box_kernel_2d[Self.dtype, Self.color_space.channels()](size))

    def gaussian_blur[border: Border](mut self, size: Int, std_dev: Optional[Float64] = None) raises:
        self.filter[border](Filters.gaussian_kernel_2d[Self.dtype, Self.color_space.channels()](size=size, std_dev=std_dev))

    def gaussian_blurred[border: Border](self, size: Int, std_dev: Optional[Float64] = None) raises -> Self:
        return self.filtered[border](Filters.gaussian_kernel_2d[Self.dtype, Self.color_space.channels()](size=size, std_dev=std_dev))

    #
    # Filtering
    #
    def filter[border: Border](mut self, kernel: Matrix[Self.dtype, Self.color_space.channels()]) raises:
        self.filtered[border](kernel).copy_into(self)

    def filtered[border: Border](self, kernel: Matrix[Self.dtype, Self.color_space.channels()]) raises -> Self:
        if kernel.rows() > self.height() or kernel.cols() > self.width():
            raise Error("Kernel must not be larger than the image along any dimension")

        var count = kernel.strided_count()
        var flipped_kernel = kernel.rotated_180()

        if count == 1:
            return self._direct_convolution[border, 1](flipped_kernel)
        elif count == 2:
            return self._direct_convolution[border, 2](flipped_kernel)
        elif count <= 4:
            return self._direct_convolution[border, 4](flipped_kernel)
        elif count <= 8:
            return self._direct_convolution[border, 8](flipped_kernel)
        elif count <= 16:
            return self._direct_convolution[border, 16](flipped_kernel)
        elif count <= 32:
            return self._direct_convolution[border, 32](flipped_kernel)
        elif count <= 64:
            return self._direct_convolution[border, 64](flipped_kernel)
        elif count <= 128:
            return self._direct_convolution[border, 128](flipped_kernel)
        else:
            return self._fourier_convolution[border](flipped_kernel)

    def _direct_convolution[border: Border, width: Int](self, kernel: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        var result = Self(height=self.height(), width=self.width())

        var half_kernel_height = kernel.rows() // 2
        var half_kernel_width = kernel.cols() // 2

        var y_offset = SIMD[DType.int, width]()
        var x_offset = SIMD[DType.int, width]()
        var offset = SIMD[DType.int, width]()
        var mask = SIMD[DType.bool, width](fill=False)
        var width_range = SIMD[DType.int, width]()

        for kernel_y in range(0, kernel.rows()):
            for kernel_x in range(0, kernel.cols()):
                var index = kernel_y * kernel.cols() + kernel_x
                y_offset[index] = kernel_y - half_kernel_height
                x_offset[index] = kernel_x - half_kernel_width
                offset[index] = (y_offset[index] * self.width() + x_offset[index]) * Self.color_space.channels()
                mask[index] = True
                width_range[index] = index

        comptime for channel in range(Self.color_space.channels()):
            var kernel_vector = kernel.strided_gather(row=0, col=0, component=channel, offset=width_range, mask=mask).value

            def process_row(y: Int) {mut result, imm}:
                var min_patch_y = y - half_kernel_height
                var max_patch_y = y + half_kernel_height

                for x in range(result.width()):
                    var min_patch_x = x - half_kernel_width
                    var max_patch_x = x + half_kernel_width

                    if min_patch_y >= 0 and max_patch_y < self.height() and min_patch_x >= 0 and max_patch_x < self.width():
                        var value = (
                            self._matrix.gather(
                                row=y,
                                col=x,
                                component=channel,
                                offset=offset,
                                mask=mask,
                            )
                            * kernel_vector
                        ).value.reduce_add()

                        result._strided_store(value, y=y, x=x, channel=channel)
                    else:
                        var pixel_sum = Scalar[Self.dtype]()

                        comptime for index in range(width):
                            if mask[index]:
                                pixel_sum = kernel_vector[index].fma(
                                    self._bordered_load[border](
                                        y=Int(y + y_offset[index]),
                                        x=Int(x + x_offset[index]),
                                        channel=channel,
                                    ),
                                    pixel_sum,
                                )

                        result._strided_store(pixel_sum, y=y, x=x, channel=channel)

            parallelize(process_row, result.height())

        return result^

    def _fourier_convolution[border: Border](self, kernel: Matrix[Self.dtype, Self.color_space.channels()]) -> Self:
        var half_kernel_rows = kernel.rows() // 2
        var half_kernel_cols = kernel.cols() // 2

        var padded_self = self.padded[border](height=half_kernel_rows, width=half_kernel_cols)
        var padded_kernel = Matrix[fft_dtype, Self.color_space.channels(), complex=True](
            rows=padded_self.height(), cols=padded_self.width()
        )

        comptime for component in range(kernel.depth):
            for row in range(kernel.rows()):
                for col in range(kernel.cols()):
                    var shifted_row = (row - half_kernel_rows) % padded_self.height()
                    var shifted_col = (col - half_kernel_cols) % padded_self.width()

                    padded_kernel._strided_store(
                        ScalarNumber[fft_dtype, complex=True](
                            real=kernel._strided_load(row=row, col=col, component=component).value.cast[fft_dtype](), imaginary=0.0
                        ),
                        row=shifted_row,
                        col=shifted_col,
                        component=component,
                    )

        var padded_spectrum = padded_self.spectrum()
        padded_spectrum *= padded_kernel.fourier_transform()

        var padded_result = padded_spectrum.fourier_transform[inverse=True]()

        var result = Self(height=self.height(), width=self.width())

        def take_real_value[width: Int](row: Int, col: Int, component: Int) {mut result, imm}:
            result._strided_store(
                padded_result._strided_load[width](row=half_kernel_rows + row, col=half_kernel_cols + col, component=component)
                .real()
                .cast[Self.dtype](),
                y=row,
                x=col,
                channel=component,
            )

        self._matrix.strided_iterate_indices(take_real_value)

        padded_result.keep()

        return result^

    @always_inline
    def _bordered_load[border: Border](self, y: Int, x: Int, channel: Int) -> Scalar[Self.dtype]:
        comptime if border == Border.zero:
            if y >= 0 and y < self.height() and x >= 0 and x < self.width():
                return self._strided_load(y=y, x=x, channel=channel)
            else:
                return Scalar[Self.dtype](0)
        elif border == Border.wrap:
            return self._strided_load(y=y % self.height(), x=x % self.width(), channel=channel)
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

            return self._strided_load(y=reflected_y, x=reflected_x, channel=channel)
        else:
            abort("Unimplemented border type in Image._bordered_load()")

    #
    # Type Conversion
    #
    def as_type[new_dtype: DType](self) -> Image[Self.color_space, new_dtype]:
        comptime if new_dtype == Self.dtype:
            return rebind[Image[Self.color_space, new_dtype]](self).copy()
        else:
            return Image[Self.color_space, new_dtype](self._matrix.as_type[new_dtype]())

    #
    # Color Space Conversion
    #
    @always_inline
    def converted[new_color_space: ColorSpace](self) -> Image[new_color_space, Self.dtype]:
        comptime if new_color_space == Self.color_space:
            return rebind[Image[new_color_space, Self.dtype]](self).copy()
        else:
            return self.converted_as_type[new_color_space, Self.dtype]()

    def converted_as_type[new_color_space: ColorSpace, new_dtype: DType](self) -> Image[new_color_space, new_dtype]:
        comptime if new_dtype == Self.dtype and new_color_space == Self.color_space:
            return rebind[Image[new_color_space, new_dtype]](self).copy()
        elif new_color_space == Self.color_space:
            return Image[new_color_space, new_dtype](self._matrix.as_type[new_dtype]().rebind[new_color_space.channels()]())
        else:
            var result = Image[new_color_space, new_dtype](height=self.height(), width=self.width())

            def convert_row(y: Int) {mut result, imm}:
                def convert_cols[width: Int](x: Int) {mut result, imm self, imm y}:
                    # Greyscale ->
                    comptime if Self.color_space == ColorSpace.greyscale:
                        var grey = self._strided_load[width](y=y, x=x, channel=0).cast[new_dtype]()

                        # RGB
                        comptime if new_color_space == ColorSpace.rgb:
                            result._strided_store(grey, y=y, x=x, channel=0)
                            result._strided_store(grey, y=y, x=x, channel=1)
                            result._strided_store(grey, y=y, x=x, channel=2)

                    # RGB ->
                    elif Self.color_space == ColorSpace.rgb:
                        var red = self._strided_load[width](y=y, x=x, channel=0).cast[DType.float64]()
                        var green = self._strided_load[width](y=y, x=x, channel=1).cast[DType.float64]()
                        var blue = self._strided_load[width](y=y, x=x, channel=2).cast[DType.float64]()

                        # Greyscale
                        comptime if new_color_space == ColorSpace.greyscale:
                            var grey = 0.299 * red + 0.587 * green + 0.114 * blue
                            result._strided_store(grey.cast[new_dtype](), y=y, x=x, channel=0)

                vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](self.width(), convert_cols)

            parallelize(convert_row, self.height())

            return result^

    #
    # Saving to File
    #
    def save[file_type: ImageFile](self, path: StringLiteral) raises:
        self.save[file_type](Path(path))

    def save[file_type: ImageFile](self, path: Path) raises:
        ImageWriter(path).write[file_type](self)

    #
    # Writable
    #
    def write_to[W: Writer](self, mut writer: W):
        writer.write(
            "[Image: height = ",
            self.height(),
            ", width = ",
            self.width(),
            ", color_space = ",
            Self.color_space,
            ", bit_depth = ",
            bit_width_of[Self.dtype](),
            "]",
        )
