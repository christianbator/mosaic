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

from mosaic.matrix import Matrix, Number, ScalarNumber
from mosaic.utilities import optimal_simd_width

from .image_reader import ImageReader
from .image_writer import ImageWriter

#
# Image
#
struct Image[dtype: DType, color_space: ColorSpace](Movable, EqualityComparable, ExplicitlyCopyable, Stringable, Writable):

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
        self._matrix = Matrix[dtype, color_space.channels()](rows = height, cols = width)

    fn __init__(out self, width: Int, height: Int, owned data: UnsafePointer[Scalar[dtype]]):
        self._matrix = Matrix[dtype, color_space.channels()](rows = height, cols = width, data = data)
    
    fn __init__(out self, owned matrix: Matrix[dtype, color_space.channels()]):
        self._matrix = matrix^
    
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
        return self._matrix.planar_count()

    @always_inline
    fn channels(self) -> Int:
        return color_space.channels()

    @always_inline
    fn samples(self) -> Int:
        return self._matrix.count()
    
    @always_inline
    fn bit_depth(self) -> Int:
        return dtype.bitwidth()
    
    @always_inline
    fn bytes(self) -> Int:
        return self.samples() * self.bit_depth() // 8
    
    #
    # Public Access
    #
    fn __getitem__(self, y: Int, x: Int) -> Scalar[dtype]:
        constrained[color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"]()

        return self._matrix[y, x, 0].value

    fn __getitem__(self, y: Int, x: Int, channel: Int) -> Scalar[dtype]:
        return self._matrix[y, x, channel].value

    fn __setitem__(mut self, y: Int, x: Int, value: Scalar[dtype]):
        constrained[color_space.channels() == 1, "Must specify channel for image in color space with channels > 1"]()

        self._matrix[y, x, 0] = value

    fn __setitem__(mut self, y: Int, x: Int, channel: Int, value: Scalar[dtype]):
        self._matrix[y, x, channel] = value

    fn strided_load[width: Int](self, y: Int, x: Int, channel: Int) -> SIMD[dtype, width]:
        return self._matrix.strided_load[width](row = y, col = x, component = channel).value

    fn strided_store[width: Int](mut self, y: Int, x: Int, channel: Int, value: SIMD[dtype, width]):
        self._matrix.strided_store[width](row = y, col = x, component = channel, number = value)
    
    fn unsafe_data_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self._matrix.unsafe_data_ptr()

    fn unsafe_uint8_ptr(self) -> UnsafePointer[UInt8]:
        return self._matrix.unsafe_uint8_ptr()

    #
    # ExplicitlyCopyable
    #
    fn copy(self) -> Self:
        return Self(matrix = self._matrix.copy())

    #
    # EqualityComparable
    #
    fn __eq__(self, other: Self) -> Bool:
        return self._matrix == other._matrix
    
    fn __ne__(self, other: Self) -> Bool:
        return not(self == other)

    #
    # Type Conversion
    #
    fn astype[new_dtype: DType](self) -> Image[new_dtype, color_space]:
        var result = Image[new_dtype, color_space](width = self.width(), height = self.height())
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
        var result = Image[new_dtype, new_color_space](width = self.width(), height = self.height())
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
                        var grey = self.strided_load[width](y = y, x = x, channel = 0).cast[new_dtype]()

                        # RGB
                        @parameter
                        if new_color_space == ColorSpace.rgb:
                            dest.strided_store(y = y, x = x, channel = 0, value = grey)
                            dest.strided_store(y = y, x = x, channel = 1, value = grey)
                            dest.strided_store(y = y, x = x, channel = 2, value = grey)

                    # RGB ->
                    elif color_space == ColorSpace.rgb:
                        var red = self.strided_load[width](y = y, x = x, channel = 0)
                        var green = self.strided_load[width](y = y, x = x, channel = 1)
                        var blue = self.strided_load[width](y = y, x = x, channel = 2)

                        # Greyscale
                        @parameter
                        if new_color_space == ColorSpace.greyscale:
                            var grey = 0.299 * red.cast[DType.float64]() + 0.587 * green.cast[DType.float64]() + 0.114 * blue.cast[DType.float64]()
                            dest.strided_store(y = y, x = x, channel = 0, value = grey.cast[new_dtype]())     

                vectorize[convert_row_pixels, Self.optimal_simd_width](self.width())
            parallelize[convert_row](num_work_items = self.height())

    #
    # Orientation
    #
    fn flip_horizontally(mut self):
        self._matrix.flip_horizontally()
    
    fn flipped_horizontally(self) -> Self:
        return Self(matrix = self._matrix.flipped_horizontally())
    
    fn flip_vertically(mut self):
        self._matrix.flip_vertically()
    
    fn flipped_vertically(self) -> Self:
        return Self(matrix = self._matrix.flipped_vertically())

    fn flip(mut self):
        self._matrix.flip()

    fn flipped(self) -> Self:
        return Self(matrix = self._matrix.flipped())

    #
    # Filtering
    #
    fn filter[border: Border](mut self, kernel: Matrix[dtype, color_space.channels()]):
        self.filtered[border](kernel)._matrix.copy_into(self._matrix)

    fn filtered[border: Border](self, kernel: Matrix[dtype, color_space.channels()]) -> Self:
        var result = Self(width = self.width(), height = self.height())
        self.filtered_into[border](dest = result, kernel = kernel)

        return result^

    fn filtered_into[border: Border](self, mut dest: Self, kernel: Matrix[dtype, color_space.channels()]):
        var count = kernel.planar_count()

        if count == 1:
            self._direct_convolution[border, 1](dest = dest, kernel = kernel.flipped())
        elif count == 2:
            self._direct_convolution[border, 2](dest = dest, kernel = kernel.flipped())
        elif count <= 4:
            self._direct_convolution[border, 4](dest = dest, kernel = kernel.flipped())
        elif count <= 8:
            self._direct_convolution[border, 8](dest = dest, kernel = kernel.flipped())
        elif count <= 16:
            self._direct_convolution[border, 16](dest = dest, kernel = kernel.flipped())
        elif count <= 32:
            self._direct_convolution[border, 32](dest = dest, kernel = kernel.flipped())
        elif count <= 64:
            self._direct_convolution[border, 64](dest = dest, kernel = kernel.flipped())
        elif count <= 128:
            self._direct_convolution[border, 128](dest = dest, kernel = kernel.flipped())
        else:
            abort("Direct convolution for kernels with plane counts greater than 512 is not supported")        

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
            # TODO: Create planar slice of kernel data and fill the kernel vector from that, or create an as_simd() method on Matrix?
            for row in range(kernel.rows()):
                for col in range(kernel.cols()):
                    kernel_vector[row * kernel.cols() + col] = kernel.strided_load[1](row = row, col = col, component = channel).value

            @parameter
            fn process_row(y: Int):
                var min_patch_y = y - half_kernel_height
                var max_patch_y = y + half_kernel_height

                for x in range(dest.width()):
                    if max_patch_y < self.height() and min_patch_y >= 0 and (x + half_kernel_width) < self.width() and (x - half_kernel_width) >= 0:
                        dest[y, x, channel] = (
                            self._matrix.gather(row = y, col = x, component = channel, offset_vector = offset_vector, mask_vector = mask_vector) * kernel_vector
                        ).value.reduce_add()
                    else:
                        var pixel_sum = Scalar[dtype]()

                        @parameter
                        for index in range(width):
                            if mask_vector[index]:
                                pixel_sum = kernel_vector[index].fma(
                                    self._bordered_load[border](y = Int(y + y_offset_vector[index]), x = Int(x + x_offset_vector[index]), channel = channel),
                                    pixel_sum
                                )
                        
                        dest[y, x, channel] = pixel_sum

            parallelize[process_row](dest.height())

    fn _filter_dft[border: Border, new_dtype: DType](self, mut dest: Image[new_dtype, color_space], flipped_kernel: Matrix[dtype, color_space.channels()]):
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
            abort()
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
        return "[Image: width = " + String(self.width()) + ", height = " + String(self.height()) + ", color_space = " + String(color_space) + ", bit_depth = " + String(dtype.bitwidth()) + "]"

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(String(self))

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
        return not(self == other)
