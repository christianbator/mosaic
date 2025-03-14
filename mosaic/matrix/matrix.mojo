#
# matrix.mojo
# mosaic
#
# Created by Christian Bator on 02/02/2025
#

from os import abort
from memory import UnsafePointer, memset_zero, memcpy
from algorithm import vectorize, parallelize
from random import rand
from math import ceildiv, isclose
from collections import InlineArray

from mosaic.utilities import optimal_simd_width

#
# Matrix
#
struct Matrix[dtype: DType, depth: Int = 1, complex: Bool = False](Movable, EqualityComparable, ExplicitlyCopyable, Stringable, Writable):

    alias optimal_simd_width = optimal_simd_width[dtype]() // 2 if complex else optimal_simd_width[dtype]()
    alias _unroll_factor = 4

    var _rows: Int
    var _cols: Int
    var _data: UnsafeNumberPointer[dtype, complex]

    #
    # Initialization
    #
    fn __init__(out self, rows: Int, cols: Int):
        self._rows = rows
        self._cols = cols
        self._data = UnsafeNumberPointer[dtype, complex](rows * cols * depth)
        
    fn __init__(out self, rows: Int, cols: Int, value: ScalarNumber[dtype, complex]):
        self._rows = rows
        self._cols = cols
        self._data = UnsafeNumberPointer[dtype, complex](rows * cols * depth)
        self.fill(value)

    fn __init__(out self, rows: Int, cols: Int, owned data: UnsafeNumberPointer[dtype, complex]):
        self._rows = rows
        self._cols = cols
        self._data = data
    
    fn __init__(out self, rows: Int, cols: Int, owned data: UnsafePointer[Scalar[dtype]]):
        self._rows = rows
        self._cols = cols
        self._data = data
    
    fn __moveinit__(out self, owned existing: Self):
        self._rows = existing._rows
        self._cols = existing._cols
        self._data = existing._data

    @staticmethod
    fn ascending(rows: Int, cols: Int) -> Self:
        var count = rows * cols * depth
        var data = UnsafeNumberPointer[dtype, complex](count)
        
        for i in range(count):
            @parameter
            if complex:
                data[i] = ScalarNumber[dtype, complex](real = i, imaginary = 0)
            else:
                data[i] = i
        
        return Self(rows = rows, cols = cols, data = data)

    @staticmethod
    fn random(rows: Int, cols: Int, min: Scalar[dtype] = Scalar[dtype].MIN_FINITE, max: Scalar[dtype] = Scalar[dtype].MAX_FINITE) -> Self:
        var data = UnsafeNumberPointer[dtype, complex](rows * cols * depth)

        var scalar_count: Int
        @parameter
        if complex:
            scalar_count = rows * cols * depth * 2
        else:
            scalar_count = rows * cols * depth
        
        rand(
            data.unsafe_ptr(),
            scalar_count,
            min = min.cast[DType.float64](),
            max = max.cast[DType.float64]()
        )

        return Self(rows = rows, cols = cols, data = data)
    
    #
    # Properties
    #
    @always_inline
    fn rows(self) -> Int:
        return self._rows

    @always_inline
    fn cols(self) -> Int:
        return self._cols

    @always_inline
    fn count(self) -> Int:
        return self._rows * self._cols * depth
    
    @always_inline
    fn planar_count(self) -> Int:
        return self._rows * self._cols

    @always_inline
    fn scalar_count(self) -> Int:
        @parameter
        if complex:
            return self.count() * 2
        else:
            return self.count()

    #
    # Public Access
    #
    fn __getitem__(self, row: Int, col: Int) -> ScalarNumber[dtype, complex]:
        constrained[depth == 1, "Must specify component for matrix with depth > 1"]()

        return self.strided_load[1](row = row, col = col, component = 0)
    
    fn __getitem__(self, row: Int, col: Int, component: Int) -> ScalarNumber[dtype, complex]:
        return self.strided_load[1](row = row, col = col, component = component)

    fn __setitem__(mut self, row: Int, col: Int, value: ScalarNumber[dtype, complex]):
        constrained[depth == 1, "Must specify component for matrix with depth > 1"]()

        self.strided_store(row = row, col = col, component = 0, value = value)

    fn __setitem__(mut self, row: Int, col: Int, component: Int, value: ScalarNumber[dtype, complex]):
        self.strided_store(row = row, col = col, component = component, value = value)

    fn strided_load[width: Int](self, row: Int, col: Int, component: Int) -> Number[dtype, complex, width]:
        return self._data.strided_load[width](
            index = self.index(row = row, col = col, component = component),
            stride = depth
        )

    fn strided_store[width: Int](mut self, row: Int, col: Int, component: Int, value: Number[dtype, complex, width]):
        self._data.strided_store(
            index = self.index(row = row, col = col, component = component),
            stride = depth,
            value = value
        )

    fn load[width: Int](self, index: Int) -> Number[dtype, complex, width]:
        return self._data.load[width](index)
    
    fn store[width: Int](mut self, index: Int, value: Number[dtype, complex, width]):
        self._data.store(index = index, value = value)

    fn gather[width: Int, //](self, row: Int, col: Int, component: Int, offset_vector: SIMD[DType.index, width], mask_vector: SIMD[DType.bool, width]) -> Number[dtype, complex, width]:
        return self._data.gather(
            index = self.index(row = row, col = col, component = component),
            offset_vector = offset_vector,
            mask_vector = mask_vector
        )
    
    fn scatter[width: Int, //](self, row: Int, col: Int, component: Int, value: Number[dtype, complex, width], offset_vector: SIMD[DType.index, width], mask_vector: SIMD[DType.bool, width]):
        self._data.scatter(
            index = self.index(row = row, col = col, component = component),
            value = value,
            offset_vector = offset_vector,
            mask_vector = mask_vector
        )

    fn unsafe_data_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self._data.unsafe_ptr()

    fn unsafe_uint8_ptr(self) -> UnsafePointer[UInt8]:
        return self._data.unsafe_ptr().bitcast[UInt8]()

    #
    # Index Utilities
    #
    @always_inline
    fn index(self, row: Int, col: Int, component: Int) -> Int:
        return (row * self._cols + col) * depth + component

    @always_inline
    fn transposed_index(self, row: Int, col: Int, component: Int) -> Int:
        return (col * self._rows + row) * depth + component

    @always_inline
    fn flattened_index(self, row: Int, offset: Int) -> Int:
        return row * self._cols * depth + offset
    
    @always_inline
    fn row_end_index(self) -> Int:
        return self._rows - 1
    
    @always_inline
    fn col_end_index(self) -> Int:
        return self._cols - 1
    
    @parameter
    fn component_end_index(self) -> Int:
        return depth - 1

    @always_inline
    fn end_index(self) -> Int:
        return self.count() - 1

    #
    # ExplicitlyCopyable
    #
    fn copy(self) -> Self:
        var result = Self(rows = self._rows, cols = self._cols)

        memcpy(dest = result._data.unsafe_ptr(), src = self._data.unsafe_ptr(), count = self.scalar_count())

        return result^

    fn copy_into(self, mut other: Self):
        memcpy(dest = other._data.unsafe_ptr(), src = self._data.unsafe_ptr(), count = self.scalar_count())

    #
    # EqualityComparable
    #
    fn __eq__(self, other: Self) -> Bool:
        if self.count() != other.count():
            return False
        
        for index in range(self.count()):
            @parameter
            if dtype.is_floating_point():
                if not isclose(self._data[index].value, other._data[index].value):
                    return False
            else:
                if self._data[index] != other._data[index]:
                    return False
        
        return True
    
    fn __ne__(self, other: Self) -> Bool:
        return not(self == other)

    #
    # Operators
    #
    fn __add__(self, rhs: ScalarNumber[dtype, complex]) -> Self:
        var result = self.copy()
        result += rhs
        return result^
    
    fn __sub__(self, rhs: ScalarNumber[dtype, complex]) -> Self:
        var result = self.copy()
        result -= rhs
        return result^

    fn __mul__(self, rhs: ScalarNumber[dtype, complex]) -> Self:
        var result = self.copy()
        result *= rhs
        return result^
    
    fn __truediv__(self, rhs: ScalarNumber[dtype, complex]) -> Self:
        var result = self.copy()
        result /= rhs
        return result^
    
    fn __floordiv__(self, rhs: ScalarNumber[dtype, complex]) -> Self:
        var result = self.copy()
        result //= rhs
        return result^
    
    fn __mod__(self, rhs: ScalarNumber[dtype, complex]) -> Self:
        var result = self.copy()
        result %= rhs
        return result^
    
    fn __pow__(self, rhs: ScalarNumber[dtype, complex]) -> Self:
        var result = self.copy()
        result **= rhs
        return result^
    
    fn __matmul__(self, rhs: Self) -> Self:
        if self._cols != rhs._rows:
            abort("Dimension mismatch for matrix multiplication: ", self._rows, "x", self._cols, " @ ", rhs._rows, "x", rhs._cols)

        var result = Self(rows = self._rows, cols = rhs._cols)

        self.matmul_into(dest = result, rhs = rhs)

        return result^
    
    #
    # In-place Operators
    #
    fn __iadd__(mut self, rhs: ScalarNumber[dtype, complex]):
        @parameter
        fn add[width: Int](value: Number[dtype, complex, width]) -> Number[dtype, complex, width]:
            return value + rhs
        
        self._for_each[add]()
    
    fn __isub__(mut self, rhs: ScalarNumber[dtype, complex]):
        @parameter
        fn sub[width: Int](value: Number[dtype, complex, width]) -> Number[dtype, complex, width]:
            return value - rhs
        
        self._for_each[sub]()

    fn __imul__(mut self, rhs: ScalarNumber[dtype, complex]):
        @parameter
        fn mul[width: Int](value: Number[dtype, complex, width]) -> Number[dtype, complex, width]:
            return value * rhs
        
        self._for_each[mul]()
    
    fn __itruediv__(mut self, rhs: ScalarNumber[dtype, complex]):
        @parameter
        fn truediv[width: Int](value: Number[dtype, complex, width]) -> Number[dtype, complex, width]:
            return value / rhs
        
        self._for_each[truediv]()
    
    fn __ifloordiv__(mut self, rhs: ScalarNumber[dtype, complex]):
        @parameter
        fn floordiv[width: Int](value: Number[dtype, complex, width]) -> Number[dtype, complex, width]:
            return value // rhs
        
        self._for_each[floordiv]()
    
    fn __imod__(mut self, rhs: ScalarNumber[dtype, complex]):
        @parameter
        fn mod[width: Int](value: Number[dtype, complex, width]) -> Number[dtype, complex, width]:
            return value % rhs
        
        self._for_each[mod]()
    
    fn __ipow__(mut self, rhs: ScalarNumber[dtype, complex]):
        @parameter
        fn pow[width: Int](value: Number[dtype, complex, width]) -> Number[dtype, complex, width]:
            return value ** rhs
        
        self._for_each[pow]()

    fn __imatmul__(mut self, other: Self):
        if self._rows != self._cols or self._rows != other._rows or self._cols != other._cols:
            abort("Dimension mismatch for in-place matrix multiplication: ", self._rows, "x", self._cols, " @ ", other._rows, "x", other._cols)

        (self @ other).copy_into(self)

    #
    # Numeric Methods
    #    
    fn matmul_into(self, mut dest: Self, rhs: Self):
        if self._cols != rhs._rows or dest._rows != self._rows or dest._cols != rhs._cols:
            abort("Dimension mismatch for matrix multiplication: ", self._rows, "x", self._cols, "@", rhs._rows, "x", rhs._cols, "->", dest._rows, "x", dest._cols)
        
        var shared_dim = self._cols

        @parameter
        for component in range(depth):
            @parameter
            fn calculate_row(row: Int):
                for k in range(shared_dim):
                    @parameter
                    fn dot_product[width: Int](col: Int):
                        
                        dest.strided_store[width](
                            row = row,
                            col = col,
                            component = component,
                            value = Number[dtype, complex, width](self[row, k, component]).fma(
                                rhs.strided_load[width](row = k, col = col, component = component),
                                dest.strided_load[width](row = row, col = col, component = component)
                            )
                            # TODO: Can I use fma()?
                            # value = dest.strided_load[width](row = row, col = col, component = component) + self[row, k, component] * rhs.strided_load[width](row = k, col = col, component = component)
                        )

                    vectorize[dot_product, Self.optimal_simd_width, unroll_factor = Self._unroll_factor](dest._cols)
            parallelize[calculate_row](dest._rows)
    
    fn _naive_matmul_into(self, mut dest: Self, rhs: Self):
        if self._cols != rhs._rows or dest._rows != self._rows or dest._cols != rhs._cols:
            abort("Dimension mismatch for matrix multiplication: ", self._rows, "x", self._cols, "@", rhs._rows, "x", rhs._cols, "->", dest._rows, "x", dest._cols)
        
        var shared_dim = self._cols

        for i in range(dest._rows):
            for j in range(dest._cols):
                for k in range(shared_dim):
                    dest[i, j] += self[i, k] * rhs[k, j]

    fn reshape(mut self, rows: Int, cols: Int):
        self._rows = rows
        self._cols = cols
    
    fn reshaped(self, rows: Int, cols: Int) -> Self:
        var result = self.copy()
        result.reshape(rows = rows, cols = cols)
        
        return result^

    fn transpose(mut self):
        if self._rows == self._cols:
            # Iterate over upper triangle of square matrix and swap elements
            @parameter
            fn iterate_over_row(row: Int):
                for col in range(row + 1, self._cols):
                    var original_index = self.index(row = row, col = col, component = 0)
                    var transposed_index = self.transposed_index(row = row, col = col, component = 0)
                    var original_element = self._load_planar_element(original_index)
                    
                    self._store_planar_element(index = original_index, element = self._load_planar_element(transposed_index))
                    self._store_planar_element(index = transposed_index, element = original_element)

            parallelize[iterate_over_row](self._rows)
        else:
            # Create a copy of the original
            var copy = self.copy()

            # Reshape self
            self.reshape(rows = self._cols, cols = self._rows)

            # Load from the copy and store in the transposed index of self
            @parameter
            fn tranpose_row(row: Int):
                for col in range(self._cols):
                    self._store_planar_element(
                        index = self.index(row = row, col = col, component = 0),
                        element = copy._load_planar_element(
                            self.transposed_index(row = row, col = col, component = 0)
                        )
                    )

            parallelize[tranpose_row](self._rows)
    
    fn transposed(self) -> Self:
        # Create a copy of the original
        var copy = self.copy()

        if self._rows == self._cols:
            copy.transpose()
        else:
            # Reshape the copy
            copy.reshape(rows = self._cols, cols = self._rows)

            # Load from self and store in the transposed index of copy
            @parameter
            fn tranpose_row(row: Int):
                for col in range(copy._cols):
                    copy._store_planar_element(
                        index = copy.index(row = row, col = col, component = 0),
                        element = self._load_planar_element(
                            copy.transposed_index(row = row, col = col, component = 0)
                        )
                    )

            parallelize[tranpose_row](copy._rows)

        return copy^

    fn flip_horizontally(mut self):
        @parameter
        fn process_row(row: Int):
            for col in range(self._cols // 2):
                var index = self.index(row = row, col = col, component = 0)
                var element = self._load_planar_element(index)
                var mirror_index = self.index(row = row, col = self.col_end_index() - col, component = 0)

                var mirror_element = self._load_planar_element(mirror_index)
                self._store_planar_element(index = index, element = mirror_element)
                self._store_planar_element(index = mirror_index, element = element)

        parallelize[process_row](self._rows)
    
    fn flipped_horizontally(self) -> Self:
        var result = self.copy()
        result.flip_horizontally()
        
        return result^
    
    fn flip_vertically(mut self):
        @parameter
        fn process_row(row: Int):
            for col in range(self._cols):
                var index = self.index(row = row, col = col, component = 0)
                var element = self._load_planar_element(index)
                var mirror_index = self.index(row = self.row_end_index() - row, col = col, component = 0)

                self._store_planar_element(index = index, element = self._load_planar_element(mirror_index))
                self._store_planar_element(index = mirror_index, element = element)

        parallelize[process_row](self._rows // 2)
    
    fn flipped_vertically(self) -> Self:
        var result = self.copy()
        result.flip_vertically()
        
        return result^

    fn flip(mut self):
        var row_range = ceildiv(self._rows, 2)

        @parameter
        fn process_row(row: Int):
            var col_range: Int
            if self._rows % 2 == 0:
                col_range = self._cols
            else:
                col_range = self._cols if row < row_range - 1 else self._cols // 2

            for col in range(col_range):
                var index = self.index(row = row, col = col, component = 0)
                var element = self._load_planar_element(index)
                var mirror_index = self.index(row = self.row_end_index() - row, col = self.col_end_index() - col, component = 0)

                self._store_planar_element(index = index, element = self._load_planar_element(mirror_index))
                self._store_planar_element(index = mirror_index, element = element)

        parallelize[process_row](row_range)

    fn flipped(self) -> Self:
        var result = self.copy()
        result.flip()

        return result^

    fn fill(mut self, scalar: ScalarNumber[dtype, complex]):
        @parameter
        fn fill[width: Int](value: Number[dtype, complex, width]) -> Number[dtype, complex, width]:
            return scalar
        
        self._for_each[fill]()

    #
    # Private Numeric Utilities
    #
    @always_inline
    fn _load_planar_element(self, index: Int) -> InlineArray[ScalarNumber[dtype, complex], depth]:
        var result = InlineArray[ScalarNumber[dtype, complex], depth](uninitialized = True)
        @parameter
        for i in range(depth):
            result[i] = self.load[1](index + i)

        return result
    
    @always_inline
    fn _store_planar_element(mut self, index: Int, element: InlineArray[ScalarNumber[dtype, complex], depth]):
        @parameter
        for i in range(depth):
            self.store(index = index + i, value = element[i])

    fn _for_each[transformer: fn[width: Int](value: Number[dtype, complex, width]) capturing -> Number[dtype, complex, width]](mut self):
        @parameter
        fn transform_row(row: Int):
            @parameter
            fn transform_flattened_elements[width: Int](flattened_element: Int):
                var index = self.flattened_index(row = row, offset = flattened_element)

                self.store(
                    index = index,
                    value = transformer[width](self.load[width](index))
                )

            vectorize[transform_flattened_elements, Self.optimal_simd_width, unroll_factor = Self._unroll_factor](self._cols * depth)
        parallelize[transform_row](self._rows)

    #
    # Type Conversion
    #
    fn astype[new_dtype: DType](self) -> Matrix[new_dtype, depth, complex]:
        var result = Matrix[new_dtype, depth, complex](rows = self._rows, cols = self._cols)
        self.astype_into(result)

        return result^
    
    fn astype_into[new_dtype: DType](self, mut dest: Matrix[new_dtype, depth, complex]):
        @parameter
        if new_dtype == dtype:
            memcpy(
                dest = dest._data.unsafe_ptr(),
                src = rebind[UnsafePointer[Scalar[new_dtype]]](self._data.unsafe_ptr()),
                count = self.scalar_count()
            )
        else:
            @parameter
            fn convert_row(row: Int):
                @parameter
                fn convert_flattened_elements[width: Int](flattened_element: Int):
                    var index = self.flattened_index(row = row, offset = flattened_element)
                    var value = self.load[width](index).cast[new_dtype]()
                    dest.store(index = index, value = value)

                vectorize[convert_flattened_elements, Self.optimal_simd_width, unroll_factor = Self._unroll_factor](self._cols * depth)
            parallelize[convert_row](self._rows)
    
    fn _unsafe_astype_into[new_dtype: DType, new_depth: Int](self, mut dest: Matrix[new_dtype, new_depth, complex]):
        constrained[new_depth == depth]()

        @parameter
        if new_dtype == dtype:
            memcpy(dest = dest._data.unsafe_ptr(), src = rebind[UnsafePointer[Scalar[new_dtype]]](self._data.unsafe_ptr()), count = self.scalar_count())
        else:
            @parameter
            fn convert_row(row: Int):
                @parameter
                fn convert_flattened_elements[width: Int](flattened_element: Int):
                    var index = self.flattened_index(row = row, offset = flattened_element)
                    var value = self.load[width](index).cast[new_dtype]()
                    dest.store(index = index, value = value)

                vectorize[convert_flattened_elements, Self.optimal_simd_width, unroll_factor = Self._unroll_factor](self._cols * depth)
            parallelize[convert_row](self._rows)
    
    #
    # Stringable & Writable
    #
    fn __str__(self) -> String:
        var result = String("[Matrix:\n  [\n")

        for row in range(self._rows):
            var row_string = String("    [")
            
            for col in range(self._cols):
                var index = self.index(row = row, col = col, component = 0)
                var suffix = ", " if col < self._cols - 1 else ""
                row_string += String(self.load[depth](index)) + suffix
            
            row_string += ("]," if row < self._rows - 1 else "]")
            result += row_string + "\n"
                    
        result += "  ]\n]"

        return result

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(String(self))
