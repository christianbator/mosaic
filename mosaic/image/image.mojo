#
# image.mojo
# mosaic
#
# Created by Christian Bator on 12/14/2024
#

from os import abort
from pathlib import Path
from memory import UnsafePointer
from algorithm import vectorize, parallelize
from collections import Optional

from mosaic.numeric import Matrix, MatrixSlice, Number, ScalarNumber, StridedRange
from mosaic.utility import optimal_simd_width, unroll_factor

from .image_reader import ImageReader
from .image_writer import ImageWriter
from .filters import Filters


#
# Image
#
struct Image[dtype: DType, color_space: ColorSpace](Movable, EqualityComparable, ExplicitlyCopyable, Stringable, Writable):
    #
    # Fields
    #
    alias optimal_simd_width = optimal_simd_width[dtype]()

    var _matrix: Matrix[dtype, color_space.channels()]

    #
    # Initialization
    #
    fn __init__(out self, path: String) raises:
        self = Self(Path(path))

    fn __init__(out self, path: Path) raises:
        self = ImageReader[dtype, color_space](path).read()

    fn __init__(out self, width: Int, height: Int):
        self._matrix = Matrix[dtype, color_space.channels()](rows=height, cols=width)

    fn __init__(
        out self,
        width: Int,
        height: Int,
        owned data: UnsafePointer[Scalar[dtype]],
    ):
        self._matrix = Matrix[dtype, color_space.channels()](rows=height, cols=width, data=data)

    fn __init__(out self, owned matrix: Matrix[dtype, color_space.channels()]):
        self._matrix = matrix^

    fn __init__(out self, single_channel_matrix: Matrix[dtype], channel: Int):
        self._matrix = Matrix[dtype, color_space.channels()](rows=single_channel_matrix.rows(), cols=single_channel_matrix.cols())

        @parameter
        fn process_row(row: Int):
            @parameter
            fn process_col[width: Int](col: Int):
                self.strided_store(y=row, x=col, channel=channel, value=single_channel_matrix.strided_load[width](row=row, col=col, component=0).value)

            vectorize[process_col, Self.optimal_simd_width, unroll_factor=unroll_factor](single_channel_matrix.cols())

        parallelize[process_row](single_channel_matrix.rows())

    fn __moveinit__(out self, owned existing: Self):
        self._matrix = existing._matrix^

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

    #
    # Public Access
    #
    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> Scalar[dtype]:
        constrained[color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"]()

        return self[y, x, 0]

    @always_inline
    fn __getitem__(self, y: Int, x: Int, channel: Int) -> Scalar[dtype]:
        return self._matrix[y, x, channel].value

    @always_inline
    fn __setitem__(mut self, y: Int, x: Int, value: Scalar[dtype]):
        constrained[color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"]()

        self[y, x, 0] = value

    @always_inline
    fn __setitem__(mut self, y: Int, x: Int, channel: Int, value: Scalar[dtype]):
        self._matrix[y, x, channel] = value

    @always_inline
    fn load[width: Int](self, y: Int, x: Int) -> SIMD[dtype, width]:
        constrained[color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"]()

        return self.strided_load[width](y=y, x=x, channel=0)

    @always_inline
    fn strided_load[width: Int](self, y: Int, x: Int, channel: Int) -> SIMD[dtype, width]:
        return self._matrix.strided_load[width](row=y, col=x, component=channel).value

    @always_inline
    fn store[width: Int](mut self, y: Int, x: Int, value: SIMD[dtype, width]):
        constrained[color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"]()

        self.strided_store(y=y, x=x, channel=0, value=value)

    @always_inline
    fn strided_store[width: Int](mut self, y: Int, x: Int, channel: Int, value: SIMD[dtype, width]):
        self._matrix.strided_store[width](row=y, col=x, component=channel, value=value)

    #
    # Unsafe Access
    #
    fn unsafe_data_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self._matrix.unsafe_data_ptr()

    fn unsafe_uint8_ptr(self) -> UnsafePointer[UInt8]:
        return self._matrix.unsafe_uint8_ptr()

    #
    # Slicing
    #
    @always_inline
    fn __getitem__[mut: Bool, origin: Origin[mut], //](ref [origin]self, y_slice: Slice, x_slice: Slice) -> ImageSlice[dtype, color_space, origin]:
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
    fn slice[mut: Bool, origin: Origin[mut], //](ref [origin]self, y_range: StridedRange) -> ImageSlice[dtype, color_space, origin]:
        return self.slice(y_range=y_range, x_range=StridedRange(self.width()))

    @always_inline
    fn slice[mut: Bool, origin: Origin[mut], //](ref [origin]self, *, x_range: StridedRange) -> ImageSlice[dtype, color_space, origin]:
        return self.slice(y_range=StridedRange(self.height()), x_range=x_range)

    @always_inline
    fn slice[mut: Bool, origin: Origin[mut], //](ref [origin]self, y_range: StridedRange, x_range: StridedRange) -> ImageSlice[dtype, color_space, origin]:
        return ImageSlice[dtype, color_space, origin](image=self, y_range=y_range, x_range=x_range)

    @always_inline
    fn channel_slice[channel: Int](self) -> MatrixSlice[StridedRange(channel, channel + 1), dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.component_slice[channel]()

    @always_inline
    fn channel_slice[
        channel: Int
    ](self, y_range: StridedRange) -> MatrixSlice[StridedRange(channel, channel + 1), dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.component_slice[channel](row_range=y_range)

    @always_inline
    fn channel_slice[
        channel: Int
    ](self, *, x_range: StridedRange) -> MatrixSlice[StridedRange(channel, channel + 1), dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.component_slice[channel](col_range=x_range)

    @always_inline
    fn channel_slice[
        channel: Int
    ](self, y_range: StridedRange, x_range: StridedRange) -> MatrixSlice[
        StridedRange(channel, channel + 1), dtype, color_space.channels(), False, __origin_of(self._matrix)
    ]:
        return self._matrix.component_slice[channel](row_range=y_range, col_range=x_range)

    @always_inline
    fn strided_slice[channel_range: StridedRange](self) -> MatrixSlice[channel_range, dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.strided_slice[channel_range]()

    @always_inline
    fn strided_slice[
        channel_range: StridedRange
    ](self, y_range: StridedRange) -> MatrixSlice[channel_range, dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.strided_slice[channel_range](row_range=y_range)

    @always_inline
    fn strided_slice[
        channel_range: StridedRange
    ](self, *, x_range: StridedRange) -> MatrixSlice[channel_range, dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.strided_slice[channel_range](col_range=x_range)

    @always_inline
    fn strided_slice[
        channel_range: StridedRange
    ](self, y_range: StridedRange, x_range: StridedRange) -> MatrixSlice[channel_range, dtype, color_space.channels(), False, __origin_of(self._matrix)]:
        return self._matrix.strided_slice[channel_range](y_range, x_range)

    @always_inline
    fn extract_channel[channel: Int](self) -> Matrix[dtype]:
        return self._matrix.extract_component[channel]()

    #
    # ExplicitlyCopyable
    #
    fn copy(self) -> Self:
        return Self(matrix=self._matrix.copy())

    #
    # EqualityComparable
    #
    fn __eq__(self, other: Self) -> Bool:
        return self._matrix == other._matrix

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    #
    # Type Conversion
    #
    fn astype[new_dtype: DType](self) -> Image[new_dtype, color_space]:
        var result = Image[new_dtype, color_space](width=self.width(), height=self.height())
        self.astype_into(result)

        return result^

    fn astype_into[new_dtype: DType](self, mut dest: Image[new_dtype, color_space]):
        self._matrix.astype_into[new_dtype](dest._matrix)

    #
    # Color Space Conversion
    #
    fn converted[new_color_space: ColorSpace](self) -> Image[dtype, new_color_space]:
        return self.converted_astype[new_color_space, dtype]()

    fn converted_into[new_color_space: ColorSpace](self, mut dest: Image[dtype, new_color_space]):
        self.converted_astype_into(dest)

    fn converted_astype[new_color_space: ColorSpace, new_dtype: DType](self) -> Image[new_dtype, new_color_space]:
        var result = Image[new_dtype, new_color_space](width=self.width(), height=self.height())
        self.converted_astype_into(result)

        return result^

    fn converted_astype_into[new_color_space: ColorSpace, new_dtype: DType](self, mut dest: Image[new_dtype, new_color_space]):
        @parameter
        if new_color_space == color_space:
            self._matrix._unsafe_astype_into[new_dtype, new_color_space.channels()](dest._matrix)
        else:

            @parameter
            fn convert_row(y: Int):
                @parameter
                fn convert_row_pixels[width: Int](x: Int):
                    # Greyscale ->
                    @parameter
                    if color_space == ColorSpace.greyscale:
                        var grey = self.strided_load[width](y=y, x=x, channel=0).cast[new_dtype]()

                        # RGB
                        @parameter
                        if new_color_space == ColorSpace.rgb:
                            dest.strided_store(y=y, x=x, channel=0, value=grey)
                            dest.strided_store(y=y, x=x, channel=1, value=grey)
                            dest.strided_store(y=y, x=x, channel=2, value=grey)

                    # RGB ->
                    elif color_space == ColorSpace.rgb:
                        var red = self.strided_load[width](y=y, x=x, channel=0)
                        var green = self.strided_load[width](y=y, x=x, channel=1)
                        var blue = self.strided_load[width](y=y, x=x, channel=2)

                        # Greyscale
                        @parameter
                        if new_color_space == ColorSpace.greyscale:
                            var grey = 0.299 * red.cast[DType.float64]() + 0.587 * green.cast[DType.float64]() + 0.114 * blue.cast[DType.float64]()
                            dest.strided_store(
                                y=y,
                                x=x,
                                channel=0,
                                value=grey.cast[new_dtype](),
                            )

                vectorize[convert_row_pixels, Self.optimal_simd_width, unroll_factor=unroll_factor](self.width())

            parallelize[convert_row](num_work_items=self.height())

    #
    # Geometric Transformations
    #
    fn flip_horizontally(mut self):
        self._matrix.flip_horizontally()

    fn flipped_horizontally(self) -> Self:
        return Self(matrix=self._matrix.flipped_horizontally())

    fn flip_vertically(mut self):
        self._matrix.flip_vertically()

    fn flipped_vertically(self) -> Self:
        return Self(matrix=self._matrix.flipped_vertically())

    fn rotate_180(mut self):
        self._matrix.rotate_180()

    fn rotated_180(self) -> Self:
        return Self(matrix=self._matrix.rotated_180())

    #
    # Common Filters
    #
    fn box_blur[border: Border](mut self, size: Int):
        self.filter[border](Filters.box_kernel_2d[dtype, color_space.channels()](size))

    fn box_blurred[border: Border](self, size: Int) -> Self:
        return self.filtered[border](Filters.box_kernel_2d[dtype, color_space.channels()](size))

    fn gaussian_blur[border: Border](mut self, size: Int, std_dev: Optional[Float64] = None):
        self.filter[border](Filters.gaussian_kernel_2d[dtype, color_space.channels()](size=size, std_dev=std_dev))

    fn gaussian_blurred[border: Border](mut self, size: Int, std_dev: Optional[Float64] = None) -> Self:
        return self.filtered[border](Filters.gaussian_kernel_2d[dtype, color_space.channels()](size=size, std_dev=std_dev))

    #
    # Filtering
    #
    fn filter[border: Border](mut self, kernel: Matrix[dtype, color_space.channels()]):
        self.filtered[border](kernel)._matrix.copy_into(self._matrix)

    fn filtered[border: Border](self, kernel: Matrix[dtype, color_space.channels()]) -> Self:
        var result = Self(width=self.width(), height=self.height())
        self.filtered_into[border](dest=result, kernel=kernel)

        return result^

    fn filtered_into[border: Border](self, mut dest: Self, kernel: Matrix[dtype, color_space.channels()]):
        var count = kernel.strided_count()

        if count == 1:
            self._direct_convolution[border, 1](dest=dest, kernel=kernel.rotated_180())
        elif count == 2:
            self._direct_convolution[border, 2](dest=dest, kernel=kernel.rotated_180())
        elif count <= 4:
            self._direct_convolution[border, 4](dest=dest, kernel=kernel.rotated_180())
        elif count <= 8:
            self._direct_convolution[border, 8](dest=dest, kernel=kernel.rotated_180())
        elif count <= 16:
            self._direct_convolution[border, 16](dest=dest, kernel=kernel.rotated_180())
        elif count <= 32:
            self._direct_convolution[border, 32](dest=dest, kernel=kernel.rotated_180())
        elif count <= 64:
            self._direct_convolution[border, 64](dest=dest, kernel=kernel.rotated_180())
        elif count <= 128:
            self._direct_convolution[border, 128](dest=dest, kernel=kernel.rotated_180())
        else:
            abort("Direct convolution for kernels with strided counts greater than 128 is not supported yet")

    fn _direct_convolution[border: Border, width: Int](self, mut dest: Self, kernel: Matrix[dtype, color_space.channels()]):
        var half_kernel_width = kernel.cols() // 2
        var half_kernel_height = kernel.rows() // 2

        var kernel_vector = SIMD[dtype, width]()
        var y_offset_vector = SIMD[DType.index, width]()
        var x_offset_vector = SIMD[DType.index, width]()
        var offset_vector = SIMD[DType.index, width]()
        var mask_vector = SIMD[DType.bool, width](False)

        for kernel_y in range(0, kernel.rows()):
            for kernel_x in range(0, kernel.cols()):
                var index = kernel_y * kernel.cols() + kernel_x
                y_offset_vector[index] = kernel_y - half_kernel_height
                x_offset_vector[index] = kernel_x - half_kernel_width
                offset_vector[index] = (y_offset_vector[index] * self.width() + x_offset_vector[index]) * color_space.channels()
                mask_vector[index] = True

        @parameter
        for channel in range(color_space.channels()):
            # TODO: Create strided slice of kernel data and fill the kernel vector from that, or create an as_simd() method on Matrix?
            for row in range(kernel.rows()):
                for col in range(kernel.cols()):
                    kernel_vector[row * kernel.cols() + col] = kernel.strided_load[1](row=row, col=col, component=channel).value

            @parameter
            fn process_row(y: Int):
                var min_patch_y = y - half_kernel_height
                var max_patch_y = y + half_kernel_height

                for x in range(dest.width()):
                    if max_patch_y < self.height() and min_patch_y >= 0 and (x + half_kernel_width) < self.width() and (x - half_kernel_width) >= 0:
                        dest[y, x, channel] = (
                            self._matrix.gather(
                                row=y,
                                col=x,
                                component=channel,
                                offset_vector=offset_vector,
                                mask_vector=mask_vector,
                            )
                            * kernel_vector
                        ).value.reduce_add()
                    else:
                        var pixel_sum = Scalar[dtype]()

                        @parameter
                        for index in range(width):
                            if mask_vector[index]:
                                pixel_sum = kernel_vector[index].fma(
                                    self._bordered_load[border](
                                        y=Int(y + y_offset_vector[index]),
                                        x=Int(x + x_offset_vector[index]),
                                        channel=channel,
                                    ),
                                    pixel_sum,
                                )

                        dest[y, x, channel] = pixel_sum

            parallelize[process_row](dest.height())

    fn _discrete_fourier_transform_convolution[
        border: Border, new_dtype: DType
    ](self, mut dest: Image[new_dtype, color_space], flipped_kernel: Matrix[dtype, color_space.channels()],):
        pass

    fn _bordered_load[border: Border](self, y: Int, x: Int, channel: Int) -> Scalar[dtype]:
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
            abort("Unimplemented border type in Image._bordered_load()")
            while True:
                pass

    #
    # Saving to File
    #
    fn save[file_type: ImageFileType](self, path: String) raises:
        self.save[file_type](Path(path))

    fn save[file_type: ImageFileType](self, path: Path) raises:
        ImageWriter(path).write[file_type](self)

    #
    # Stringable & Writable
    #
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[Image: width = ", self.width(), ", height = ", self.height(), ", color_space = ", color_space, ", bit_depth = ", dtype.bitwidth(), "]")


#
# ImagePointer
#
@value
struct ImagePointer[dtype: DType, color_space: ColorSpace](EqualityComparable):
    var _pointer: Pointer[Image[dtype, color_space], ImmutableAnyOrigin]

    fn __init__(out self, image: Image[dtype, color_space]):
        self._pointer = rebind[Pointer[Image[dtype, color_space], ImmutableAnyOrigin]](Pointer.address_of(image))

    fn __getitem__(self) -> ref [self._pointer[]] Image[dtype, color_space]:
        return self._pointer[]

    fn __eq__(self, other: Self) -> Bool:
        return self._pointer == other._pointer

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)
