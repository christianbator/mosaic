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

from mosaic.utility import optimal_simd_width


#
# Matrix
#
struct Matrix[dtype: DType, depth: Int = 1, complex: Bool = False](Movable, EqualityComparable, ExplicitlyCopyable, Stringable, Writable):
    alias optimal_simd_width = optimal_simd_width[dtype]() // 2 if complex else optimal_simd_width[dtype]()
    alias _unroll_factor = 4

    var _rows: Int
    var _cols: Int
    var _data: UnsafeNumberPointer[dtype, complex=complex]

    #
    # Initialization
    #
    fn __init__(out self, rows: Int, cols: Int):
        self._rows = rows
        self._cols = cols
        self._data = UnsafeNumberPointer[dtype, complex=complex](rows * cols * depth)

    fn __init__(
        out self,
        rows: Int,
        cols: Int,
        value: ScalarNumber[dtype, complex=complex],
    ):
        self._rows = rows
        self._cols = cols
        self._data = UnsafeNumberPointer[dtype, complex=complex](rows * cols * depth)
        self.fill(value)

    fn __init__(
        out self,
        rows: Int,
        cols: Int,
        *values: ScalarNumber[dtype, complex=complex],
    ):
        # TODO: Make this a compile-time check when possible
        debug_assert(
            rows * cols * depth == len(values),
            "mismatch in the number of values in the Matrix variadic constructor",
        )

        self._rows = rows
        self._cols = cols
        self._data = UnsafeNumberPointer[dtype, complex=complex](rows * cols * depth)

        for index in range(len(values)):
            self._store(index=index, value=values[index])

    fn __init__(
        out self,
        rows: Int,
        cols: Int,
        owned data: UnsafeNumberPointer[dtype, complex=complex],
    ):
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
        var data = UnsafeNumberPointer[dtype, complex=complex](count)

        for i in range(count):

            @parameter
            if complex:
                data[i] = ScalarNumber[dtype, complex=complex](real=i, imaginary=0)
            else:
                data[i] = i

        return Self(rows=rows, cols=cols, data=data)

    @staticmethod
    fn random(
        rows: Int,
        cols: Int,
        min: Scalar[dtype] = Scalar[dtype].MIN_FINITE,
        max: Scalar[dtype] = Scalar[dtype].MAX_FINITE,
    ) -> Self:
        var data = UnsafeNumberPointer[dtype, complex=complex](rows * cols * depth)

        var scalar_count: Int

        @parameter
        if complex:
            scalar_count = rows * cols * depth * 2
        else:
            scalar_count = rows * cols * depth

        rand(
            data.unsafe_ptr(),
            scalar_count,
            min=min.cast[DType.float64](),
            max=max.cast[DType.float64](),
        )

        return Self(rows=rows, cols=cols, data=data)

    @staticmethod
    fn strided_replication(rows: Int, cols: Int, *values: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = Self(rows=rows, cols=cols)

        for row in range(rows):
            for col in range(cols):
                result.store_full_depth(
                    row=row,
                    col=col,
                    value=result.create_full_depth_value(values[row * cols + col]),
                )

        return result^

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
    fn components(self) -> Int:
        return depth

    @always_inline
    fn count(self) -> Int:
        return self._rows * self._cols * depth

    @always_inline
    fn strided_count(self) -> Int:
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
    fn __getitem__(self, row: Int, col: Int) -> ScalarNumber[dtype, complex=complex]:
        constrained[depth == 1, "Must specify component for matrix with depth > 1"]()

        return self.strided_load[1](row=row, col=col, component=0)

    fn __getitem__(self, row: Int, col: Int, component: Int) -> ScalarNumber[dtype, complex=complex]:
        return self.strided_load[1](row=row, col=col, component=component)

    fn __setitem__(
        mut self,
        row: Int,
        col: Int,
        value: ScalarNumber[dtype, complex=complex],
    ):
        constrained[depth == 1, "Must specify component for matrix with depth > 1"]()

        self.strided_store(row=row, col=col, component=0, value=value)

    fn __setitem__(
        mut self,
        row: Int,
        col: Int,
        component: Int,
        value: ScalarNumber[dtype, complex=complex],
    ):
        self.strided_store(row=row, col=col, component=component, value=value)

    fn strided_load[width: Int](self, row: Int, col: Int, component: Int) -> Number[dtype, width, complex=complex]:
        return self._data.strided_load[width](
            index=self.index(row=row, col=col, component=component),
            stride=depth,
        )

    fn strided_store[
        width: Int, //
    ](mut self, row: Int, col: Int, component: Int, value: Number[dtype, width, complex=complex],):
        self._data.strided_store(
            index=self.index(row=row, col=col, component=component),
            stride=depth,
            value=value,
        )

    fn gather[
        width: Int, //
    ](
        self,
        row: Int,
        col: Int,
        component: Int,
        offset_vector: SIMD[DType.index, width],
        mask_vector: SIMD[DType.bool, width],
    ) -> Number[
        dtype, width, complex=complex
    ]:
        return self._data.gather(
            index=self.index(row=row, col=col, component=component),
            offset_vector=offset_vector,
            mask_vector=mask_vector,
        )

    fn scatter[
        width: Int, //
    ](
        self,
        row: Int,
        col: Int,
        component: Int,
        value: Number[dtype, width, complex=complex],
        offset_vector: SIMD[DType.index, width],
        mask_vector: SIMD[DType.bool, width],
    ):
        self._data.scatter(
            index=self.index(row=row, col=col, component=component),
            value=value,
            offset_vector=offset_vector,
            mask_vector=mask_vector,
        )

    #
    # Slicing
    #
    fn __getitem__[
        mut: Bool, origin: Origin[mut], //
    ](ref [origin]self, row_slice: Slice, col_slice: Slice) -> MatrixSlice[StridedRange(depth), dtype, depth, complex, origin]:
        return MatrixSlice[StridedRange(depth), dtype, depth, complex, origin](
            matrix=self,
            row_range=StridedRange(
                slice=row_slice,
                default_start=0,
                default_end=self._rows,
                default_step=1,
            ),
            col_range=StridedRange(
                slice=col_slice,
                default_start=0,
                default_end=self._cols,
                default_step=1,
            ),
        )

    fn component_slice[
        component: Int, /, mut: Bool, origin: Origin[mut]
    ](ref [origin]self) -> MatrixSlice[StridedRange(component, component + 1), dtype, depth, complex, origin]:
        return self.strided_slice[StridedRange(component, component + 1)]()

    fn component_slice[
        component: Int, /, mut: Bool, origin: Origin[mut]
    ](ref [origin]self, row_range: StridedRange) -> MatrixSlice[StridedRange(component, component + 1), dtype, depth, complex, origin]:
        return self.strided_slice[StridedRange(component, component + 1)](row_range)

    fn component_slice[
        component: Int, /, mut: Bool, origin: Origin[mut]
    ](ref [origin]self, *, col_range: StridedRange) -> MatrixSlice[StridedRange(component, component + 1), dtype, depth, complex, origin]:
        return self.strided_slice[StridedRange(component, component + 1)](col_range=col_range)

    fn component_slice[
        component: Int, /, mut: Bool, origin: Origin[mut]
    ](ref [origin]self, row_range: StridedRange, col_range: StridedRange) -> MatrixSlice[StridedRange(component, component + 1), dtype, depth, complex, origin]:
        return self.strided_slice[StridedRange(component, component + 1)](row_range=row_range, col_range=col_range)

    fn strided_slice[
        component_range: StridedRange, /, mut: Bool, origin: Origin[mut]
    ](ref [origin]self) -> MatrixSlice[component_range, dtype, depth, complex, origin]:
        return self.strided_slice[component_range](row_range=(0, self._rows), col_range=(0, self._cols))

    fn strided_slice[
        component_range: StridedRange, /, mut: Bool, origin: Origin[mut]
    ](ref [origin]self, row_range: StridedRange) -> MatrixSlice[component_range, dtype, depth, complex, origin]:
        return self.strided_slice[component_range](row_range=row_range, col_range=(0, self._cols))

    fn strided_slice[
        component_range: StridedRange, /, mut: Bool, origin: Origin[mut]
    ](ref [origin]self, *, col_range: StridedRange) -> MatrixSlice[component_range, dtype, depth, complex, origin]:
        return self.strided_slice[component_range](row_range=(0, self._rows), col_range=col_range)

    fn strided_slice[
        component_range: StridedRange, /, mut: Bool, origin: Origin[mut]
    ](ref [origin]self, row_range: StridedRange, col_range: StridedRange) -> MatrixSlice[component_range, dtype, depth, complex, origin]:
        return MatrixSlice[component_range, dtype, depth, complex, origin](matrix=self, row_range=row_range, col_range=col_range)

    #
    # Private Access
    #
    fn _load[width: Int](self, index: Int) -> Number[dtype, width, complex=complex]:
        return self._data.load[width](index)

    fn _store[width: Int, //](mut self, index: Int, value: Number[dtype, width, complex=complex]):
        self._data.store(index=index, value=value)

    #
    # Unsafe Access
    #
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
        var result = Self(rows=self._rows, cols=self._cols)

        memcpy(
            dest=result._data.unsafe_ptr(),
            src=self._data.unsafe_ptr(),
            count=self.scalar_count(),
        )

        return result^

    fn copy_into(self, mut other: Self):
        memcpy(
            dest=other._data.unsafe_ptr(),
            src=self._data.unsafe_ptr(),
            count=self.scalar_count(),
        )

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
        return not (self == other)

    #
    # Operators
    #
    fn __add__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result += rhs
        return result^

    fn __sub__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result -= rhs
        return result^

    fn __mul__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result *= rhs
        return result^

    fn __truediv__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result /= rhs
        return result^

    fn __floordiv__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result //= rhs
        return result^

    fn __mod__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result %= rhs
        return result^

    fn __pow__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result **= rhs
        return result^

    fn __matmul__(self, rhs: Self) -> Self:
        if self._cols != rhs._rows:
            abort(
                "Dimension mismatch for matrix multiplication: ",
                self._rows,
                "x",
                self._cols,
                " @ ",
                rhs._rows,
                "x",
                rhs._cols,
            )

        var result = Self(rows=self._rows, cols=rhs._cols)

        self.matmul_into(dest=result, rhs=rhs)

        return result^

    #
    # In-place Operators
    #
    fn __iadd__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn add[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value + rhs

        self.for_each[add]()

    fn __isub__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn sub[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value - rhs

        self.for_each[sub]()

    fn __imul__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn mul[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value * rhs

        self.for_each[mul]()

    fn __itruediv__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn truediv[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value / rhs

        self.for_each[truediv]()

    fn __ifloordiv__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn floordiv[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value // rhs

        self.for_each[floordiv]()

    fn __imod__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn mod[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value % rhs

        self.for_each[mod]()

    fn __ipow__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn pow[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value**rhs

        self.for_each[pow]()

    fn __imatmul__(mut self, other: Self):
        if self._rows != self._cols or self._rows != other._rows or self._cols != other._cols:
            abort(
                "Dimension mismatch for in-place matrix multiplication: ",
                self._rows,
                "x",
                self._cols,
                " @ ",
                other._rows,
                "x",
                other._cols,
            )

        (self @ other).copy_into(self)

    #
    # Numeric Methods
    #
    fn strided_sum(self, component: Int) -> ScalarNumber[dtype, complex=complex]:
        var result = ScalarNumber[dtype, complex=complex](0)

        for row in range(self._rows):

            @parameter
            fn process_cols[width: Int](col: Int):
                result += self.strided_load[width](row=row, col=col, component=component).reduce_add()

            vectorize[
                process_cols,
                Self.optimal_simd_width,
                unroll_factor = Self._unroll_factor,
            ](self._cols)

        return result

    fn sum(self: Matrix[dtype, 1, complex=complex]) -> ScalarNumber[dtype, complex=complex]:
        return self.strided_sum(0)

    fn strided_average(self, component: Int) -> ScalarNumber[DType.float64, complex=complex]:
        return self.strided_sum(component).cast[DType.float64]() / self.strided_count()

    fn average(self: Matrix[dtype, 1, complex=complex]) -> ScalarNumber[DType.float64, complex=complex]:
        return self.strided_average(0)

    fn strided_min(self: Matrix[dtype, depth, False], component: Int) -> Scalar[dtype]:
        var result = Scalar[dtype].MAX_FINITE

        for row in range(self._rows):

            @parameter
            fn process_cols[width: Int](col: Int):
                result = min(
                    result,
                    self.strided_load[width](row=row, col=col, component=component).reduce_min().value,
                )

            vectorize[
                process_cols,
                Self.optimal_simd_width,
                unroll_factor = Self._unroll_factor,
            ](self._cols)

        return result

    fn min(self: Matrix[dtype, 1, False]) -> Scalar[dtype]:
        return self.strided_min(0)

    fn strided_max(self: Matrix[dtype, depth, False], component: Int) -> Scalar[dtype]:
        var result = Scalar[dtype].MIN_FINITE

        for row in range(self._rows):

            @parameter
            fn process_cols[width: Int](col: Int):
                result = max(
                    result,
                    self.strided_load[width](row=row, col=col, component=component).reduce_max().value,
                )

            vectorize[
                process_cols,
                Self.optimal_simd_width,
                unroll_factor = Self._unroll_factor,
            ](self._cols)

        return result

    fn max(self: Matrix[dtype, 1, False]) -> Scalar[dtype]:
        return self.strided_max(0)

    fn strided_normalize(mut self):
        @parameter
        for component in range(depth):
            var strided_sum = self.strided_sum(component)

            @parameter
            @__copy_capture(strided_sum)
            fn normalize[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
                return value / strided_sum

            self.strided_for_each[normalize](component)

    fn matmul_into(self, mut dest: Self, rhs: Self):
        if self._cols != rhs._rows or dest._rows != self._rows or dest._cols != rhs._cols:
            abort(
                "Dimension mismatch for matrix multiplication: ",
                self._rows,
                "x",
                self._cols,
                "@",
                rhs._rows,
                "x",
                rhs._cols,
                "->",
                dest._rows,
                "x",
                dest._cols,
            )

        var shared_dim = self._cols

        @parameter
        for component in range(depth):

            @parameter
            fn calculate_row(row: Int):
                for k in range(shared_dim):

                    @parameter
                    fn dot_product[width: Int](col: Int):
                        dest.strided_store[width](
                            row=row,
                            col=col,
                            component=component,
                            value=Number[dtype, width, complex=complex](self[row, k, component]).fma(
                                rhs.strided_load[width](row=k, col=col, component=component),
                                dest.strided_load[width](row=row, col=col, component=component),
                            )
                            # TODO: Can I use fma()?
                            # value = dest.strided_load[width](row = row, col = col, component = component) + self[row, k, component] * rhs.strided_load[width](row = k, col = col, component = component)
                        )

                    vectorize[
                        dot_product,
                        Self.optimal_simd_width,
                        unroll_factor = Self._unroll_factor,
                    ](dest._cols)

            parallelize[calculate_row](dest._rows)

    fn _naive_matmul_into(self, mut dest: Self, rhs: Self):
        if self._cols != rhs._rows or dest._rows != self._rows or dest._cols != rhs._cols:
            abort(
                "Dimension mismatch for matrix multiplication: ",
                self._rows,
                "x",
                self._cols,
                "@",
                rhs._rows,
                "x",
                rhs._cols,
                "->",
                dest._rows,
                "x",
                dest._cols,
            )

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
        result.reshape(rows=rows, cols=cols)

        return result^

    fn transpose(mut self):
        if self._rows == self._cols:
            # Iterate over upper triangle of square matrix and swap elements
            @parameter
            fn iterate_over_row(row: Int):
                for col in range(row + 1, self._cols):
                    var original_value = self.load_full_depth(row=row, col=col)
                    self.store_full_depth(
                        row=row,
                        col=col,
                        value=self.load_full_depth(row=col, col=row),
                    )
                    self.store_full_depth(row=col, col=row, value=original_value)

            parallelize[iterate_over_row](self._rows)
        else:
            # Create a copy of the original
            var copy = self.copy()

            # Reshape self
            self.reshape(rows=self._cols, cols=self._rows)

            # Load from the copy and store in the transposed index of self
            @parameter
            fn tranpose_row(row: Int):
                for col in range(self._cols):
                    self.store_full_depth(
                        row=row,
                        col=col,
                        value=copy.load_full_depth(row=col, col=row),
                    )

            parallelize[tranpose_row](self._rows)

    fn transposed(self) -> Self:
        # Create a copy of the original
        var copy = self.copy()

        if self._rows == self._cols:
            copy.transpose()
        else:
            # Reshape the copy
            copy.reshape(rows=self._cols, cols=self._rows)

            # Load from self and store in the transposed index of copy
            @parameter
            fn tranpose_row(row: Int):
                for col in range(copy._cols):
                    copy.store_full_depth(
                        row=row,
                        col=col,
                        value=self.load_full_depth(row=col, col=row),
                    )

            parallelize[tranpose_row](copy._rows)

        return copy^

    fn flip_horizontally(mut self):
        @parameter
        fn process_row(row: Int):
            for col in range(self._cols // 2):
                var original_value = self.load_full_depth(row=row, col=col)
                self.store_full_depth(
                    row=row,
                    col=col,
                    value=self.load_full_depth(row=row, col=self.col_end_index() - col),
                )
                self.store_full_depth(
                    row=row,
                    col=self.col_end_index() - col,
                    value=original_value,
                )

        parallelize[process_row](self._rows)

    fn flipped_horizontally(self) -> Self:
        var result = self.copy()
        result.flip_horizontally()

        return result^

    fn flip_vertically(mut self):
        @parameter
        fn process_row(row: Int):
            for col in range(self._cols):
                var original_value = self.load_full_depth(row=row, col=col)
                self.store_full_depth(
                    row=row,
                    col=col,
                    value=self.load_full_depth(row=self.row_end_index() - row, col=col),
                )
                self.store_full_depth(
                    row=self.row_end_index() - row,
                    col=col,
                    value=original_value,
                )

        parallelize[process_row](self._rows // 2)

    fn flipped_vertically(self) -> Self:
        var result = self.copy()
        result.flip_vertically()

        return result^

    fn rotate_180(mut self):
        var row_range = ceildiv(self._rows, 2)

        @parameter
        fn process_row(row: Int):
            var col_range: Int
            if self._rows % 2 == 0:
                col_range = self._cols
            else:
                col_range = self._cols if row < row_range - 1 else self._cols // 2

            for col in range(col_range):
                var original_value = self.load_full_depth(row=row, col=col)
                self.store_full_depth(
                    row=row,
                    col=col,
                    value=self.load_full_depth(
                        row=self.row_end_index() - row,
                        col=self.col_end_index() - col,
                    ),
                )
                self.store_full_depth(
                    row=self.row_end_index() - row,
                    col=self.col_end_index() - col,
                    value=original_value,
                )

        parallelize[process_row](row_range)

    fn rotated_180(self) -> Self:
        var result = self.copy()
        result.rotate_180()

        return result^

    fn fill(mut self, scalar: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn fill[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return scalar

        self.for_each[fill]()

    fn strided_for_each[
        transformer: fn[width: Int] (value: Number[dtype, width, complex=complex]) capturing -> Number[dtype, width, complex=complex]
    ](mut self, component: Int):
        @parameter
        fn transform_row(row: Int):
            @parameter
            fn transform_col[width: Int](col: Int):
                self.strided_store(
                    row=row,
                    col=col,
                    component=component,
                    value=transformer[width](value=self.strided_load[width](row=row, col=col, component=component)),
                )

            vectorize[
                transform_col,
                Self.optimal_simd_width,
                unroll_factor = Self._unroll_factor,
            ](self._cols)

        parallelize[transform_row](self._rows)

    fn for_each[transformer: fn[width: Int] (value: Number[dtype, width, complex=complex]) capturing -> Number[dtype, width, complex=complex]](mut self):
        @parameter
        fn transform_row(row: Int):
            @parameter
            fn transform_flattened_elements[width: Int](flattened_element: Int):
                var index = self.flattened_index(row=row, offset=flattened_element)

                self._store(
                    index=index,
                    value=transformer[width](self._load[width](index)),
                )

            vectorize[
                transform_flattened_elements,
                Self.optimal_simd_width,
                unroll_factor = Self._unroll_factor,
            ](self._cols * depth)

        parallelize[transform_row](self._rows)

    fn load_full_depth(self, row: Int, col: Int) -> InlineArray[ScalarNumber[dtype, complex=complex], depth]:
        var result = InlineArray[ScalarNumber[dtype, complex=complex], depth](uninitialized=True)

        @parameter
        for component in range(depth):
            result[component] = self.strided_load[1](row=row, col=col, component=component)

        return result

    fn store_full_depth(
        mut self,
        row: Int,
        col: Int,
        value: InlineArray[ScalarNumber[dtype, complex=complex], depth],
    ):
        @parameter
        for component in range(depth):
            self.strided_store(row=row, col=col, component=component, value=value[component])

    fn create_full_depth_value(self, value: ScalarNumber[dtype, complex=complex]) -> InlineArray[ScalarNumber[dtype, complex=complex], depth]:
        var result = InlineArray[ScalarNumber[dtype, complex=complex], depth](uninitialized=True)

        @parameter
        for component in range(depth):
            result[component] = value

        return result

    fn create_full_depth_value(self, *values: ScalarNumber[dtype, complex=complex]) -> InlineArray[ScalarNumber[dtype, complex=complex], depth]:
        # TODO: Make this a compile-time check when possible
        debug_assert(
            depth == len(values),
            "mismatch in the number of values in the full depth element variadic constructor",
        )

        var result = InlineArray[ScalarNumber[dtype, complex=complex], depth](uninitialized=True)

        @parameter
        for component in range(depth):
            result[component] = values[component]

        return result

    #
    # Type Conversion
    #
    fn astype[new_dtype: DType](self) -> Matrix[new_dtype, depth, complex=complex]:
        var result = Matrix[new_dtype, depth, complex=complex](rows=self._rows, cols=self._cols)
        self.astype_into(result)

        return result^

    fn astype_into[new_dtype: DType](self, mut dest: Matrix[new_dtype, depth, complex=complex]):
        @parameter
        if new_dtype == dtype:
            memcpy(
                dest=dest._data.unsafe_ptr(),
                src=rebind[UnsafePointer[Scalar[new_dtype]]](self._data.unsafe_ptr()),
                count=self.scalar_count(),
            )
        else:

            @parameter
            fn convert_row(row: Int):
                @parameter
                fn convert_flattened_elements[width: Int](flattened_element: Int):
                    var index = self.flattened_index(row=row, offset=flattened_element)
                    var value = self._load[width](index).cast[new_dtype]()
                    dest._store(index=index, value=value)

                vectorize[
                    convert_flattened_elements,
                    Self.optimal_simd_width,
                    unroll_factor = Self._unroll_factor,
                ](self._cols * depth)

            parallelize[convert_row](self._rows)

    fn _unsafe_astype_into[new_dtype: DType, new_depth: Int](self, mut dest: Matrix[new_dtype, new_depth, complex=complex]):
        constrained[new_depth == depth]()

        @parameter
        if new_dtype == dtype:
            memcpy(
                dest=dest._data.unsafe_ptr(),
                src=rebind[UnsafePointer[Scalar[new_dtype]]](self._data.unsafe_ptr()),
                count=self.scalar_count(),
            )
        else:

            @parameter
            fn convert_row(row: Int):
                @parameter
                fn convert_flattened_elements[width: Int](flattened_element: Int):
                    var index = self.flattened_index(row=row, offset=flattened_element)
                    var value = self._load[width](index).cast[new_dtype]()
                    dest._store(index=index, value=value)

                vectorize[
                    convert_flattened_elements,
                    Self.optimal_simd_width,
                    unroll_factor = Self._unroll_factor,
                ](self._cols * depth)

            parallelize[convert_row](self._rows)

    #
    # Stringable & Writable
    #
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[Matrix:\n  [\n")

        for row in range(self._rows):
            writer.write("    [")
            for col in range(self._cols):

                @parameter
                if depth > 1:
                    writer.write("[")

                @parameter
                for component in range(depth):
                    writer.write(self[row, col, component])

                    @parameter
                    if depth > 1:
                        writer.write(", " if component < depth - 1 else "]")
                writer.write(", " if col < self._cols - 1 else "")
            writer.write("],\n" if row < self._rows - 1 else "]\n")
        writer.write("  ]\n]")
