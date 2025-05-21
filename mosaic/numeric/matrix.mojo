#
# matrix.mojo
# mosaic
#
# Created by Christian Bator on 02/02/2025
#

from os import abort
from memory import UnsafePointer, memset_zero, memcpy
from algorithm import vectorize, parallelize
from math import cos, sin, pi, floor, ceil, trunc, ceildiv, isclose, Ceilable, CeilDivable, Floorable, Truncable
from collections import InlineArray

from mosaic.utility import optimal_simd_width, unroll_factor, _assert

from .fft import fft, fft_dtype


#
# Matrix
#
struct Matrix[dtype: DType, depth: Int = 1, *, complex: Bool = False](
    Absable, Ceilable, CeilDivable, EqualityComparable, ExplicitlyCopyable, Floorable, Movable, Roundable, Stringable, Truncable, Writable
):
    #
    # Fields
    #
    alias optimal_simd_width = optimal_simd_width[dtype]() // 2 if complex else optimal_simd_width[dtype]()

    var _rows: Int
    var _cols: Int
    var _data: NumericArray[dtype, complex=complex]

    #
    # Initialization
    #
    fn __init__(out self, *, rows: Int, cols: Int):
        constrained[depth > 0]()
        _assert(rows > 0 and cols > 0, "Rows and cols must be greather than 0")

        self._rows = rows
        self._cols = cols
        self._data = NumericArray[dtype, complex=complex](count=rows * cols * depth)

    fn __init__(out self, *, rows: Int, cols: Int, value: ScalarNumber[dtype, complex=complex]):
        constrained[depth > 0]()
        _assert(rows > 0 and cols > 0, "Rows and cols must be greather than 0")

        self._rows = rows
        self._cols = cols
        self._data = NumericArray[dtype, complex=complex](count=rows * cols * depth)
        self.fill(value)

    fn __init__(out self, *, rows: Int, cols: Int, owned values: List[ScalarNumber[dtype, complex=complex]]):
        constrained[depth > 0]()
        _assert(rows * cols * depth == len(values), "Mismatch in list length for Matrix constructor")

        self._rows = rows
        self._cols = cols
        self._data = NumericArray[dtype, complex=complex](values.steal_data(), count=rows * cols * depth)

    fn __init__(out self, *, rows: Int, cols: Int, owned data: NumericArray[dtype, complex=complex]):
        constrained[depth > 0]()
        _assert(rows * cols * depth == len(data), "Mismatch in data length for Matrix constructor")

        self._rows = rows
        self._cols = cols
        self._data = data^

    # This is an unsafe convenience constructor
    fn __init__(out self, rows: Int, cols: Int, owned data: UnsafePointer[Scalar[dtype]]):
        constrained[depth > 0]()
        _assert(rows > 0 and cols > 0, "Rows and cols must be greather than 0")

        self._rows = rows
        self._cols = cols
        self._data = NumericArray[dtype, complex=complex](data, count=rows * cols * depth)

    fn __moveinit__(out self, owned existing: Self):
        self._rows = existing._rows
        self._cols = existing._cols
        self._data = existing._data^

    @staticmethod
    fn ascending(*, rows: Int, cols: Int) -> Self:
        return Self(rows=rows, cols=cols, data=NumericArray[dtype, complex=complex].ascending(count=rows * cols * depth))

    @staticmethod
    fn random(*, rows: Int, cols: Int, min: Scalar[dtype] = Scalar[dtype].MIN_FINITE, max: Scalar[dtype] = Scalar[dtype].MAX_FINITE) -> Self:
        return Self(rows=rows, cols=cols, data=NumericArray[dtype, complex=complex].random(count=rows * cols * depth))

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
    fn _scalar_count(self) -> Int:
        @parameter
        if complex:
            return 2 * self.count()
        else:
            return self.count()

    @parameter
    fn _scalar_depth(self) -> Int:
        if complex:
            return 2 * depth
        else:
            return depth

    #
    # Public Access
    #
    @always_inline
    fn __getitem__(self: Matrix[dtype, 1, complex=complex], row: Int, col: Int) raises -> ScalarNumber[dtype, complex=complex]:
        return self[row, col, 0]

    @always_inline
    fn __getitem__(self, row: Int, col: Int, component: Int) raises -> ScalarNumber[dtype, complex=complex]:
        return self.strided_load(row=row, col=col, component=component)

    @always_inline
    fn __setitem__(mut self: Matrix[dtype, 1, complex=complex], row: Int, col: Int, value: ScalarNumber[dtype, complex=complex]) raises:
        self[row, col, 0] = value

    @always_inline
    fn __setitem__(mut self, row: Int, col: Int, component: Int, value: ScalarNumber[dtype, complex=complex]) raises:
        self.strided_store(value, row=row, col=col, component=component)

    @always_inline
    fn strided_load[width: Int = 1](self: Matrix[dtype, 1, complex=complex], row: Int, col: Int) raises -> Number[dtype, width, complex=complex]:
        return self.strided_load[width](row=row, col=col, component=0)

    @always_inline
    fn strided_load[width: Int = 1](self, row: Int, col: Int, component: Int) raises -> Number[dtype, width, complex=complex]:
        return self._data.strided_load[width](index=self.index(row=row, col=col, component=component), stride=depth)

    @always_inline
    fn strided_store[width: Int, //](mut self: Matrix[dtype, 1, complex=complex], value: Number[dtype, width, complex=complex], row: Int, col: Int) raises:
        self.strided_store(value, row=row, col=col, component=0)

    @always_inline
    fn strided_store[width: Int, //](mut self, value: Number[dtype, width, complex=complex], row: Int, col: Int, component: Int) raises:
        self._data.strided_store(value, index=self.index(row=row, col=col, component=component), stride=depth)

    @always_inline
    fn gather[
        width: Int, //
    ](self, row: Int, col: Int, component: Int, offset: SIMD[DType.index, width], mask: SIMD[DType.bool, width],) -> Number[dtype, width, complex=complex]:
        return self._data.gather(index=self.index(row=row, col=col, component=component), offset=offset, mask=mask)

    @always_inline
    fn strided_gather[
        width: Int, //
    ](self, row: Int, col: Int, component: Int, offset: SIMD[DType.index, width], mask: SIMD[DType.bool, width],) -> Number[dtype, width, complex=complex]:
        return self.gather(row=row, col=col, component=component, offset=depth * offset, mask=mask)

    @always_inline
    fn scatter[
        width: Int, //
    ](self, value: Number[dtype, width, complex=complex], row: Int, col: Int, component: Int, offset: SIMD[DType.index, width], mask: SIMD[DType.bool, width],):
        self._data.scatter(value, index=self.index(row=row, col=col, component=component), offset=offset, mask=mask)

    @always_inline
    fn strided_scatter[
        width: Int, //
    ](self, row: Int, col: Int, component: Int, value: Number[dtype, width, complex=complex], offset: SIMD[DType.index, width], mask: SIMD[DType.bool, width],):
        self.scatter(value, row=row, col=col, component=component, offset=depth * offset, mask=mask)

    fn load_full_depth(self, row: Int, col: Int) raises -> InlineArray[ScalarNumber[dtype, complex=complex], depth]:
        var result = InlineArray[ScalarNumber[dtype, complex=complex], depth](uninitialized=True)

        @parameter
        for component in range(depth):
            result[component] = self.strided_load(row=row, col=col, component=component)

        return result

    fn store_full_depth(mut self, value: InlineArray[ScalarNumber[dtype, complex=complex], depth], row: Int, col: Int) raises:
        @parameter
        for component in range(depth):
            self.strided_store(value[component], row=row, col=col, component=component)

    fn create_full_depth_value(self, value: ScalarNumber[dtype, complex=complex]) -> InlineArray[ScalarNumber[dtype, complex=complex], depth]:
        var result = InlineArray[ScalarNumber[dtype, complex=complex], depth](uninitialized=True)

        @parameter
        for component in range(depth):
            result[component] = value

        return result

    fn create_full_depth_value(self, *values: ScalarNumber[dtype, complex=complex]) -> InlineArray[ScalarNumber[dtype, complex=complex], depth]:
        # TODO: Make this a compile-time check when possible
        debug_assert(depth == len(values), "Mismatch in the number of values in the full depth value variadic constructor")

        var result = InlineArray[ScalarNumber[dtype, complex=complex], depth](uninitialized=True)

        @parameter
        for component in range(depth):
            result[component] = values[component]

        return result

    fn strided_iterate[handler: fn[width: Int] (value: Number[dtype, width, complex=complex], row: Int, col: Int, component: Int) capturing -> None](self):
        @parameter
        for component in range(depth):

            @parameter
            fn visit_row(row: Int):
                @parameter
                fn visit_col[width: Int](col: Int):
                    handler[width](value=self._strided_load[width](row=row, col=col, component=component), row=row, col=col, component=component)

                vectorize[
                    visit_col,
                    Self.optimal_simd_width,
                    unroll_factor=unroll_factor,
                ](self._cols)

            parallelize[visit_row](self._rows)

    fn strided_iterate_indices[handler: fn[width: Int] (row: Int, col: Int, component: Int) capturing -> None](self):
        @parameter
        for component in range(depth):

            @parameter
            fn visit_row(row: Int):
                @parameter
                fn visit_col[width: Int](col: Int):
                    handler[width](row=row, col=col, component=component)

                vectorize[
                    visit_col,
                    Self.optimal_simd_width,
                    unroll_factor=unroll_factor,
                ](self._cols)

            parallelize[visit_row](self._rows)

    fn real(self: Matrix[dtype, depth, complex=True]) -> Matrix[dtype, depth, complex=False]:
        var result = Matrix[dtype, depth, complex=False](rows=self._rows, cols=self._cols)

        @parameter
        fn take_real_value[width: Int](value: Number[dtype, width, complex=True], row: Int, col: Int, component: Int):
            result._strided_store(Number[dtype, width, complex=False](value.real()), row=row, col=col, component=component)

        self.strided_iterate[take_real_value]()

        return result^

    fn imaginary(self: Matrix[dtype, depth, complex=True]) -> Matrix[dtype, depth, complex=False]:
        var result = Matrix[dtype, depth, complex=False](rows=self._rows, cols=self._cols)

        @parameter
        fn take_imaginary_value[width: Int](value: Number[dtype, width, complex=True], row: Int, col: Int, component: Int):
            result._strided_store(Number[dtype, width, complex=False](value.imaginary()), row=row, col=col, component=component)

        self.strided_iterate[take_imaginary_value]()

        return result^

    #
    # Private Access
    #
    @always_inline
    fn _strided_load[width: Int = 1](self: Matrix[dtype, 1, complex=complex], row: Int, col: Int) -> Number[dtype, width, complex=complex]:
        return self._strided_load[width](row=row, col=col, component=0)

    @always_inline
    fn _strided_load[width: Int = 1](self, row: Int, col: Int, component: Int) -> Number[dtype, width, complex=complex]:
        return self._data.strided_load[width](index=self.index(row=row, col=col, component=component), stride=depth)

    @always_inline
    fn _strided_store[width: Int, //](mut self: Matrix[dtype, 1, complex=complex], value: Number[dtype, width, complex=complex], row: Int, col: Int):
        self._strided_store(value, row=row, col=col, component=0)

    @always_inline
    fn _strided_store[width: Int, //](mut self, value: Number[dtype, width, complex=complex], row: Int, col: Int, component: Int):
        self._data.strided_store(value, index=self.index(row=row, col=col, component=component), stride=depth)

    fn _load_full_depth(self, row: Int, col: Int) -> InlineArray[ScalarNumber[dtype, complex=complex], depth]:
        var result = InlineArray[ScalarNumber[dtype, complex=complex], depth](uninitialized=True)

        @parameter
        for component in range(depth):
            result[component] = self._strided_load(row=row, col=col, component=component)

        return result

    fn _store_full_depth(mut self, value: InlineArray[ScalarNumber[dtype, complex=complex], depth], row: Int, col: Int):
        @parameter
        for component in range(depth):
            self._strided_store(value[component], row=row, col=col, component=component)

    @always_inline
    fn _load[width: Int](self, index: Int) -> Number[dtype, width, complex=complex]:
        return self._data.load[width](index)

    @always_inline
    fn _store[width: Int, //](mut self, value: Number[dtype, width, complex=complex], index: Int):
        self._data.store(value, index=index)

    #
    # Unsafe Access
    #
    @always_inline
    fn unsafe_data_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self._data.unsafe_data_ptr()

    @always_inline
    fn unsafe_uint8_ptr(self) -> UnsafePointer[UInt8]:
        return self._data.unsafe_uint8_ptr()

    fn keep(self):
        pass

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
    # Slicing
    #
    @always_inline
    fn __getitem__[
        mut: Bool, origin: Origin[mut], //
    ](ref [origin]self, row: Int, col_slice: Slice) raises -> MatrixSlice[StridedRange(depth), dtype, depth, complex, origin]:
        return self[row : row + 1, col_slice]

    @always_inline
    fn __getitem__[
        mut: Bool, origin: Origin[mut], //
    ](ref [origin]self, row_slice: Slice, col: Int) raises -> MatrixSlice[StridedRange(depth), dtype, depth, complex, origin]:
        return self[row_slice, col : col + 1]

    @always_inline
    fn __getitem__[
        mut: Bool, origin: Origin[mut], //
    ](ref [origin]self, row_slice: Slice, col_slice: Slice) raises -> MatrixSlice[StridedRange(depth), dtype, depth, complex, origin]:
        return self.slice(
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

    @always_inline
    fn slice[
        mut: Bool, //, origin: Origin[mut]
    ](ref [origin]self, row_range: StridedRange) raises -> MatrixSlice[StridedRange(depth), dtype, depth, complex, origin]:
        return self.slice(row_range=row_range, col_range=StridedRange(self._cols))

    @always_inline
    fn slice[
        mut: Bool, //, origin: Origin[mut]
    ](ref [origin]self, *, col_range: StridedRange) raises -> MatrixSlice[StridedRange(depth), dtype, depth, complex, origin]:
        return self.slice(row_range=StridedRange(self._rows), col_range=col_range)

    @always_inline
    fn slice[
        mut: Bool, //, origin: Origin[mut]
    ](ref [origin]self, row_range: StridedRange, col_range: StridedRange) raises -> MatrixSlice[StridedRange(depth), dtype, depth, complex, origin]:
        return MatrixSlice[StridedRange(depth), dtype, depth, complex, origin](self, row_range=row_range, col_range=col_range)

    @always_inline
    fn component_slice[
        component: Int, mut: Bool, origin: Origin[mut]
    ](ref [origin]self) -> MatrixSlice[StridedRange(component, component + 1), dtype, depth, complex, origin]:
        return MatrixSlice[StridedRange(component, component + 1), dtype, depth, complex, origin](self)

    @always_inline
    fn component_slice[
        component: Int, mut: Bool, origin: Origin[mut]
    ](ref [origin]self, row_range: StridedRange) raises -> MatrixSlice[StridedRange(component, component + 1), dtype, depth, complex, origin]:
        return self.strided_slice[StridedRange(component, component + 1)](row_range)

    @always_inline
    fn component_slice[
        component: Int, mut: Bool, origin: Origin[mut]
    ](ref [origin]self, *, col_range: StridedRange) raises -> MatrixSlice[StridedRange(component, component + 1), dtype, depth, complex, origin]:
        return self.strided_slice[StridedRange(component, component + 1)](col_range=col_range)

    @always_inline
    fn component_slice[
        component: Int, mut: Bool, origin: Origin[mut]
    ](ref [origin]self, row_range: StridedRange, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(component, component + 1), dtype, depth, complex, origin
    ]:
        return self.strided_slice[StridedRange(component, component + 1)](row_range=row_range, col_range=col_range)

    @always_inline
    fn strided_slice[
        component_range: StridedRange, mut: Bool, origin: Origin[mut]
    ](ref [origin]self) -> MatrixSlice[component_range, dtype, depth, complex, origin]:
        return MatrixSlice[component_range, dtype, depth, complex, origin](self)

    @always_inline
    fn strided_slice[
        component_range: StridedRange, mut: Bool, origin: Origin[mut]
    ](ref [origin]self, row_range: StridedRange) raises -> MatrixSlice[component_range, dtype, depth, complex, origin]:
        return self.strided_slice[component_range](row_range=row_range, col_range=StridedRange(self._cols))

    @always_inline
    fn strided_slice[
        component_range: StridedRange, mut: Bool, origin: Origin[mut]
    ](ref [origin]self, *, col_range: StridedRange) raises -> MatrixSlice[component_range, dtype, depth, complex, origin]:
        return self.strided_slice[component_range](row_range=StridedRange(self._rows), col_range=col_range)

    @always_inline
    fn strided_slice[
        component_range: StridedRange, mut: Bool, origin: Origin[mut]
    ](ref [origin]self, row_range: StridedRange, col_range: StridedRange) raises -> MatrixSlice[component_range, dtype, depth, complex, origin]:
        return MatrixSlice[component_range, dtype, depth, complex, origin](self, row_range=row_range, col_range=col_range)

    fn store_sub_matrix(mut self, value: Self, row: Int, col: Int, component: Int = 0) raises:
        self.store_sub_matrix(value[:, :], row=row, col=col, component=component)

    fn store_sub_matrix[
        component_range: StridedRange, //
    ](mut self, matrix_slice: MatrixSlice[component_range, dtype, complex=complex], row: Int, col: Int, component: Int = 0) raises:
        if (component + component_range.count()) > depth or (matrix_slice.row_range().end > self._rows) or (matrix_slice.col_range().end > self._cols):
            raise Error("Out of bounds sub-matrix store")

        @parameter
        for matrix_slice_component in range(component_range.count()):

            @parameter
            fn store_row(matrix_slice_row: Int):
                @parameter
                fn store_cols[width: Int](matrix_slice_col: Int):
                    self._strided_store(
                        matrix_slice._strided_load[width](row=matrix_slice_row, col=matrix_slice_col, component=matrix_slice_component),
                        row=row + matrix_slice_row,
                        col=col + matrix_slice_col,
                        component=component + matrix_slice_component,
                    )

                vectorize[store_cols, Self.optimal_simd_width, unroll_factor=unroll_factor](matrix_slice.cols())

            parallelize[store_row](matrix_slice.rows())

    fn strided_replication[new_depth: Int](self: Matrix[dtype, 1, complex=complex]) -> Matrix[dtype, new_depth, complex=complex]:
        constrained[new_depth > depth, "Strided replication requires a desired depth > 1"]()

        var result = Matrix[dtype, new_depth, complex=complex](rows=self._rows, cols=self._cols)

        @parameter
        fn process_row(row: Int):
            for col in range(self._cols):
                result._store_full_depth(result.create_full_depth_value(self._strided_load(row=row, col=col)), row=row, col=col)

        parallelize[process_row](self._rows)

        return result^

    @staticmethod
    fn strided_replication(*, rows: Int, cols: Int, values: List[ScalarNumber[dtype, complex=complex]]) -> Self:
        _assert(rows * cols == len(values), "Mismatch in list length for Matrix strided replication constructor")

        var result = Self(rows=rows, cols=cols)

        @parameter
        fn process_row(row: Int):
            for col in range(cols):
                result._store_full_depth(result.create_full_depth_value(values[row * cols + col]), row=row, col=col)

        parallelize[process_row](rows)

        return result^

    fn copied_to_component[component: Int, new_depth: Int](self: Matrix[dtype, 1, complex=complex]) -> Matrix[dtype, new_depth, complex=complex]:
        constrained[0 <= component < new_depth, "Component must be within range of depth for copied_to_component()"]()

        var result = Matrix[dtype, new_depth, complex=complex](rows=self._rows, cols=self._cols)

        @parameter
        fn process_row(row: Int):
            @parameter
            fn process_col[width: Int](col: Int):
                result._strided_store(self._strided_load[width](row=row, col=col), row=row, col=col, component=component)

            vectorize[process_col, Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols)

        parallelize[process_row](self._rows)

        return result^

    #
    # ExplicitlyCopyable
    #
    fn copy(self) -> Self:
        return Self(rows=self._rows, cols=self._cols, data=self._data.copy())

    fn copy_into(self, mut other: Self):
        # This is bounds checked by NumericArray
        self._data.copy_into(other._data)

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
    # Operators (Scalar)
    #
    fn __neg__(self) -> Self:
        return self * -1

    fn __add__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result += rhs
        return result^

    fn __iadd__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn add[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value + rhs

        self.for_each[add]()

    fn __sub__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result -= rhs
        return result^

    fn __isub__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn sub[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value - rhs

        self.for_each[sub]()

    fn __mul__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result *= rhs
        return result^

    fn __rmul__(self, lhs: ScalarNumber[dtype, complex=complex]) -> Self:
        return self * lhs

    fn __imul__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn mul[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value * rhs

        self.for_each[mul]()

    fn __truediv__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result /= rhs
        return result^

    fn __itruediv__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn truediv[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value / rhs

        self.for_each[truediv]()

    fn __floordiv__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result //= rhs
        return result^

    fn __ifloordiv__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn floordiv[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value // rhs

        self.for_each[floordiv]()

    fn __mod__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result %= rhs
        return result^

    fn __imod__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn mod[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value % rhs

        self.for_each[mod]()

    fn __pow__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result **= rhs
        return result^

    fn __ipow__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn pow[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value**rhs

        self.for_each[pow]()

    fn __and__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result &= rhs
        return result^

    fn __iand__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn _and[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value & rhs

        self.for_each[_and]()

    fn __or__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result |= rhs
        return result^

    fn __ior__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn _or[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value | rhs

        self.for_each[_or]()

    fn __xor__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result ^= rhs
        return result^

    fn __ixor__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn _xor[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value ^ rhs

        self.for_each[_xor]()

    fn __lshift__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result <<= rhs
        return result^

    fn __ilshift__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn lshift[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value << rhs

        self.for_each[lshift]()

    fn __rshift__(self, rhs: ScalarNumber[dtype, complex=complex]) -> Self:
        var result = self.copy()
        result >>= rhs
        return result^

    fn __irshift__(mut self, rhs: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn rshift[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value >> rhs

        self.for_each[rshift]()

    #
    # Operators (Matrix)
    #
    fn __add__(self, rhs: Self) -> Self:
        var result = self.copy()
        result += rhs
        return result^

    fn __iadd__(mut self, rhs: Self):
        @parameter
        fn add[width: Int](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value + rhs

        self.for_each_zipped[add](rhs)

    fn __sub__(self, rhs: Self) -> Self:
        var result = self.copy()
        result -= rhs
        return result^

    fn __isub__(mut self, rhs: Self):
        @parameter
        fn sub[width: Int](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value - rhs

        self.for_each_zipped[sub](rhs)

    fn __mul__(self, rhs: Self) -> Self:
        var result = self.copy()
        result *= rhs
        return result^

    fn __imul__(mut self, rhs: Self):
        @parameter
        fn mul[width: Int](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value * rhs

        self.for_each_zipped[mul](rhs)

    fn __matmul__(self, rhs: Self) -> Self:
        _assert(self._cols == rhs._rows, "Dimension mismatch for matrix multiplication: ", self._rows, "x", self._cols, " @ ", rhs._rows, "x", rhs._cols)

        var result = Self(rows=self._rows, cols=rhs._cols)

        var shared_dim = self._cols

        @parameter
        for component in range(depth):

            @parameter
            fn calculate_row(row: Int):
                for k in range(shared_dim):

                    @parameter
                    fn dot_product[width: Int](col: Int):
                        result._strided_store[width](
                            row=row,
                            col=col,
                            component=component,
                            value=Number[dtype, width, complex=complex](self._strided_load(row=row, col=k, component=component)).fma(
                                rhs._strided_load[width](row=k, col=col, component=component),
                                result._strided_load[width](row=row, col=col, component=component),
                            ),
                        )

                    vectorize[
                        dot_product,
                        Self.optimal_simd_width,
                        unroll_factor=unroll_factor,
                    ](result._cols)

            parallelize[calculate_row](result._rows)

        return result^

    fn __imatmul__(mut self, other: Self):
        _assert(self._rows == self._cols and self._rows == other._rows and self._cols == other._cols, "Dimension mismatch for in-place matrix multiplication")

        (self @ other).copy_into(self)

    fn __truediv__(self, rhs: Self) -> Self:
        var result = self.copy()
        result /= rhs
        return result^

    fn __itruediv__(mut self, rhs: Self):
        @parameter
        fn truediv[
            width: Int
        ](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value / rhs

        self.for_each_zipped[truediv](rhs)

    fn __floordiv__(self, rhs: Self) -> Self:
        var result = self.copy()
        result //= rhs
        return result^

    fn __ifloordiv__(mut self, rhs: Self):
        @parameter
        fn floordiv[
            width: Int
        ](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value // rhs

        self.for_each_zipped[floordiv](rhs)

    fn __mod__(self, rhs: Self) -> Self:
        var result = self.copy()
        result %= rhs
        return result^

    fn __imod__(mut self, rhs: Self):
        @parameter
        fn mod[width: Int](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value % rhs

        self.for_each_zipped[mod](rhs)

    fn __pow__(self, rhs: Self) -> Self:
        var result = self.copy()
        result **= rhs
        return result^

    fn __ipow__(mut self, rhs: Self):
        @parameter
        fn pow[width: Int](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value**rhs

        self.for_each_zipped[pow](rhs)

    fn __and__(self, rhs: Self) -> Self:
        var result = self.copy()
        result &= rhs
        return result^

    fn __iand__(mut self, rhs: Self):
        @parameter
        fn _and[width: Int](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value & rhs

        self.for_each_zipped[_and](rhs)

    fn __or__(self, rhs: Self) -> Self:
        var result = self.copy()
        result |= rhs
        return result^

    fn __ior__(mut self, rhs: Self):
        @parameter
        fn _or[width: Int](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value | rhs

        self.for_each_zipped[_or](rhs)

    fn __xor__(self, rhs: Self) -> Self:
        var result = self.copy()
        result ^= rhs
        return result^

    fn __ixor__(mut self, rhs: Self):
        @parameter
        fn _xor[width: Int](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value ^ rhs

        self.for_each_zipped[_xor](rhs)

    fn __lshift__(self, rhs: Self) -> Self:
        var result = self.copy()
        result <<= rhs
        return result^

    fn __ilshift__(mut self, rhs: Self):
        @parameter
        fn lshift[
            width: Int
        ](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value << rhs

        self.for_each_zipped[lshift](rhs)

    fn __rshift__(self, rhs: Self) -> Self:
        var result = self.copy()
        result >>= rhs
        return result^

    fn __irshift__(mut self, rhs: Self):
        @parameter
        fn rshift[
            width: Int
        ](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value >> rhs

        self.for_each_zipped[rshift](rhs)

    #
    # Numeric Traits
    #
    fn __floor__(self) -> Self:
        var result = self.copy()

        @parameter
        fn _floor[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return floor(value)

        result.for_each[_floor]()
        return result^

    fn __ceil__(self) -> Self:
        var result = self.copy()

        @parameter
        fn _ceil[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return ceil(value)

        result.for_each[_ceil]()
        return result^

    fn __ceildiv__(self, rhs: Self) -> Self:
        var result = self.copy()

        @parameter
        fn _ceildiv[
            width: Int
        ](value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return ceildiv(value, rhs)

        result.for_each_zipped[_ceildiv](rhs)

        return result^

    fn __round__(self) -> Self:
        var result = self.copy()

        @parameter
        fn _round[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return round(value)

        result.for_each[_round]()
        return result^

    fn __round__(self, ndigits: Int) -> Self:
        var result = self.copy()

        @parameter
        fn _round[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return round(value, ndigits=ndigits)

        result.for_each[_round]()
        return result^

    fn __trunc__(self) -> Self:
        var result = self.copy()

        @parameter
        fn _trunc[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return trunc(value)

        result.for_each[_trunc]()
        return result^

    fn __abs__(self) -> Self:
        var result = self.copy()

        @parameter
        fn _abs[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return abs(value)

        result.for_each[_abs]()
        return result^

    fn norm(self: Matrix[dtype, depth, complex=True]) -> Matrix[dtype, depth, complex=False]:
        var result = Matrix[dtype, depth, complex=False](rows=self._rows, cols=self._cols)

        @parameter
        fn transform_row(row: Int):
            @parameter
            fn transform_flattened_elements[width: Int](flattened_element: Int):
                var index = self.flattened_index(row=row, offset=flattened_element)
                result._store(self._load[width](index).norm(), index=index)

            vectorize[
                transform_flattened_elements,
                Self.optimal_simd_width,
                unroll_factor=unroll_factor,
            ](self._cols * depth)

        parallelize[transform_row](self._rows)

        return result^

    #
    # Numeric Methods
    #
    fn log(self) -> Self:
        var result = self.copy()

        @parameter
        fn log[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value.log()

        result.for_each[log]()
        return result^

    fn clamp(mut self, lower_bound: ScalarNumber[dtype], upper_bound: ScalarNumber[dtype]):
        @parameter
        fn transformer[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return value.clamp(lower_bound=lower_bound, upper_bound=upper_bound)

        self.for_each[transformer]()

    fn clamped(self, lower_bound: ScalarNumber[dtype], upper_bound: ScalarNumber[dtype]) -> Self:
        var result = self.copy()
        result.clamp(lower_bound=lower_bound, upper_bound=upper_bound)

        return result^

    fn strided_sum(self, component: Int) -> ScalarNumber[dtype, complex=complex]:
        var result = ScalarNumber[dtype, complex=complex](0)

        for row in range(self._rows):

            @parameter
            fn process_cols[width: Int](col: Int):
                result += self._strided_load[width](row=row, col=col, component=component).reduce_add()

            vectorize[process_cols, Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols)

        return result

    fn strided_average(self, component: Int) -> ScalarNumber[DType.float64, complex=complex]:
        return self.strided_sum(component).cast[DType.float64]() / self.strided_count()

    fn strided_min(self: Matrix[dtype, depth], component: Int) -> ScalarNumber[dtype]:
        var result = Scalar[dtype].MAX_FINITE

        for row in range(self._rows):

            @parameter
            fn process_cols[width: Int](col: Int):
                result = min(result, self._strided_load[width](row=row, col=col, component=component).reduce_min().value)

            vectorize[process_cols, Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols)

        return result

    fn strided_max(self: Matrix[dtype, depth], component: Int) -> ScalarNumber[dtype]:
        var result = Scalar[dtype].MIN_FINITE

        for row in range(self._rows):

            @parameter
            fn process_cols[width: Int](col: Int):
                result = max(result, self._strided_load[width](row=row, col=col, component=component).reduce_max().value)

            vectorize[process_cols, Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols)

        return result

    fn strided_normalize(mut self):
        @parameter
        for component in range(depth):
            var strided_sum = self.strided_sum(component)

            @parameter
            @__copy_capture(strided_sum)
            fn normalize[width: Int](value: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
                return value / strided_sum

            self.strided_for_each[normalize](component)

    fn strided_fill(mut self, value: ScalarNumber[dtype, complex=complex], component: Int):
        @parameter
        fn fill_row(row: Int):
            @parameter
            fn fill_cols[width: Int](col: Int):
                self._strided_store(Number[dtype, width, complex=complex](value), row=row, col=col, component=component)

            vectorize[fill_cols, Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols)

        parallelize[fill_row](self._rows)

    fn fill(mut self, value: ScalarNumber[dtype, complex=complex]):
        @parameter
        fn fill_row(row: Int):
            @parameter
            fn fill_flattened_elements[width: Int](flattened_element: Int):
                self._store(Number[dtype, width, complex=complex](value), index=self.flattened_index(row=row, offset=flattened_element))

            vectorize[fill_flattened_elements, Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols * depth)

        parallelize[fill_row](self._rows)

    fn strided_for_each[
        transformer: fn[width: Int] (value: Number[dtype, width, complex=complex]) capturing -> Number[dtype, width, complex=complex]
    ](mut self, component: Int):
        @parameter
        fn transform_row(row: Int):
            @parameter
            fn transform_col[width: Int](col: Int):
                self._strided_store(
                    transformer[width](value=self._strided_load[width](row=row, col=col, component=component)), row=row, col=col, component=component
                )

            vectorize[
                transform_col,
                Self.optimal_simd_width,
                unroll_factor=unroll_factor,
            ](self._cols)

        parallelize[transform_row](self._rows)

    fn strided_for_each_zipped[
        transformer: fn[width: Int] (value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) capturing -> Number[
            dtype, width, complex=complex
        ]
    ](mut self, other: Self, component: Int):
        _assert(self._rows == other._rows and self._cols == other._cols, "Cannot perform elementwise transformation on two matrices of different sizes")

        @parameter
        fn transform_row(row: Int):
            @parameter
            fn transform_cols[width: Int](col: Int):
                self._strided_store(
                    transformer[width](
                        value=self._strided_load[width](row=row, col=col, component=component),
                        rhs=other._strided_load[width](row=row, col=col, component=component),
                    ),
                    row=row,
                    col=col,
                    component=component,
                )

            vectorize[transform_cols, Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols)

        parallelize[transform_row](self._rows)

    fn for_each[transformer: fn[width: Int] (value: Number[dtype, width, complex=complex]) capturing -> Number[dtype, width, complex=complex]](mut self):
        @parameter
        fn transform_row(row: Int):
            @parameter
            fn transform_flattened_elements[width: Int](flattened_element: Int):
                var index = self.flattened_index(row=row, offset=flattened_element)
                self._store(transformer[width](self._load[width](index)), index=index)

            vectorize[
                transform_flattened_elements,
                Self.optimal_simd_width,
                unroll_factor=unroll_factor,
            ](self._cols * depth)

        parallelize[transform_row](self._rows)

    fn for_each_zipped[
        transformer: fn[width: Int] (value: Number[dtype, width, complex=complex], rhs: Number[dtype, width, complex=complex]) capturing -> Number[
            dtype, width, complex=complex
        ]
    ](mut self, other: Self):
        _assert(self._rows == other._rows and self._cols == other._cols, "Cannot perform elementwise transformation on two matrices of different sizes")

        @parameter
        fn transform_row(row: Int):
            @parameter
            fn transform_flattened_elements[width: Int](flattened_element: Int):
                var index = self.flattened_index(row=row, offset=flattened_element)
                self._store(transformer[width](value=self._load[width](index), rhs=other._load[width](index)), index=index)

            vectorize[
                transform_flattened_elements,
                Self.optimal_simd_width,
                unroll_factor=unroll_factor,
            ](self._cols * depth)

        parallelize[transform_row](self._rows)

    fn map_to_range(mut self: Matrix[dtype, depth, complex=False], min: ScalarNumber[dtype], max: ScalarNumber[dtype]):
        @parameter
        for component in range(depth):
            var current_min = self.strided_min(component)
            var current_max = self.strided_max(component)
            var interpolation_factor = (max - min) / (current_max - current_min)

            @parameter
            @__copy_capture(current_min, interpolation_factor)
            fn map[width: Int](value: Number[dtype, width, complex=False]) -> Number[dtype, width, complex=False]:
                return min + interpolation_factor * (value - current_min)

            self.strided_for_each[map](component)

    fn mapped_to_range(self: Matrix[dtype, depth, complex=False], min: ScalarNumber[dtype], max: ScalarNumber[dtype]) -> Matrix[dtype, depth, complex=False]:
        var result = self.copy()
        result.map_to_range(min, max)

        return result^

    fn fourier_transform[*, inverse: Bool = False](self) -> Matrix[fft_dtype, depth, complex=True]:
        return fft[inverse=inverse](self)

    #
    # Geometric Transformations
    #
    fn reshape(mut self, rows: Int, cols: Int):
        _assert(rows * cols == self.strided_count(), "Cannot reshape Matrix of strided_count = ", self.strided_count(), " to rows = ", rows, ", cols = ", cols)

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
                    var original_value = self._load_full_depth(row=row, col=col)
                    self._store_full_depth(self._load_full_depth(row=col, col=row), row=row, col=col)
                    self._store_full_depth(original_value, row=col, col=row)

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
                    self._store_full_depth(copy._load_full_depth(row=col, col=row), row=row, col=col)

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
                    copy._store_full_depth(self._load_full_depth(row=col, col=row), row=row, col=col)

            parallelize[tranpose_row](copy._rows)

        return copy^

    fn flip_horizontally(mut self):
        @parameter
        fn process_row(row: Int):
            for col in range(self._cols // 2):
                var original_value = self._load_full_depth(row=row, col=col)
                self._store_full_depth(self._load_full_depth(row=row, col=self.col_end_index() - col), row=row, col=col)
                self._store_full_depth(original_value, row=row, col=self.col_end_index() - col)

        parallelize[process_row](self._rows)

    fn flipped_horizontally(self) -> Self:
        var result = self.copy()
        result.flip_horizontally()

        return result^

    fn flip_vertically(mut self):
        @parameter
        fn process_row(row: Int):
            for col in range(self._cols):
                var original_value = self._load_full_depth(row=row, col=col)
                self._store_full_depth(self._load_full_depth(row=self.row_end_index() - row, col=col), row=row, col=col)
                self._store_full_depth(original_value, row=self.row_end_index() - row, col=col)

        parallelize[process_row](self._rows // 2)

    fn flipped_vertically(self) -> Self:
        var result = self.copy()
        result.flip_vertically()

        return result^

    fn rotate_90[*, clockwise: Bool](mut self):
        self.transpose()

        @parameter
        if clockwise:
            self.flip_horizontally()
        else:
            self.flip_vertically()

    fn rotated_90[*, clockwise: Bool](self) -> Self:
        var result = self.transposed()

        @parameter
        if clockwise:
            result.flip_horizontally()
        else:
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
                var original_value = self._load_full_depth(row=row, col=col)
                self._store_full_depth(self._load_full_depth(row=self.row_end_index() - row, col=self.col_end_index() - col), row=row, col=col)
                self._store_full_depth(original_value, row=self.row_end_index() - row, col=self.col_end_index() - col)

        parallelize[process_row](row_range)

    fn rotated_180(self) -> Self:
        var result = self.copy()
        result.rotate_180()

        return result^

    fn shift_origin_to_center(mut self):
        try:
            var half_rows = self._rows // 2
            var half_cols = self._cols // 2
            var ceil_half_rows = ceildiv(self._rows, 2)
            var ceil_half_cols = ceildiv(self._cols, 2)

            var copy = self.copy()

            var top_left = copy[0:ceil_half_rows, 0:ceil_half_cols]
            var top_right = copy[0:ceil_half_rows, ceil_half_cols : self._cols]
            var bottom_left = copy[ceil_half_rows : self._rows, 0:ceil_half_cols]
            var bottom_right = copy[ceil_half_rows : self._rows, ceil_half_cols : self._cols]

            self.store_sub_matrix(top_left, row=half_rows, col=half_cols)
            self.store_sub_matrix(top_right, row=half_rows, col=0)
            self.store_sub_matrix(bottom_left, row=0, col=half_cols)
            self.store_sub_matrix(bottom_right, row=0, col=0)

        except error:
            abort(error)

    fn shifted_origin_to_center(self) -> Self:
        var result = self.copy()
        result.shift_origin_to_center()

        return result^

    fn shift_center_to_origin(mut self):
        try:
            var half_rows = self._rows // 2
            var half_cols = self._cols // 2
            var ceil_half_rows = ceildiv(self._rows, 2)
            var ceil_half_cols = ceildiv(self._cols, 2)

            var copy = self.copy()

            var top_left = copy[0:half_rows, 0:half_cols]
            var top_right = copy[0:half_rows, half_cols : self._cols]
            var bottom_left = copy[half_rows : self._rows, 0:half_cols]
            var bottom_right = copy[half_rows : self._rows, half_cols : self._cols]

            self.store_sub_matrix(top_left, row=ceil_half_rows, col=ceil_half_cols)
            self.store_sub_matrix(top_right, row=ceil_half_rows, col=0)
            self.store_sub_matrix(bottom_left, row=0, col=ceil_half_cols)
            self.store_sub_matrix(bottom_right, row=0, col=0)

        except error:
            abort(error)

    fn shifted_center_to_origin(self) -> Self:
        var result = self.copy()
        result.shift_center_to_origin()

        return result^

    fn padded(self, size: Int) -> Self:
        return self.padded(rows=size, cols=size)

    fn padded(self, rows: Int, cols: Int) -> Self:
        _assert(rows >= 0 and cols >= 0, "Must specify rows >= 0 and cols >= 0 for Matrix.padded()")

        var src_data_ptr = self.unsafe_data_ptr()

        var new_rows = self._rows + 2 * rows
        var new_cols = self._cols + 2 * cols

        var result = Self(rows=new_rows, cols=new_cols)
        var result_base_data_ptr = result.unsafe_data_ptr().offset((rows * new_cols + cols) * self._scalar_depth())

        var row_offset = self._cols * self._scalar_depth()
        var new_row_offset = new_cols * self._scalar_depth()

        @parameter
        fn copy_row(row: Int):
            memcpy(dest=result_base_data_ptr.offset(row * new_row_offset), src=src_data_ptr.offset(row * row_offset), count=row_offset)

        parallelize[copy_row](self._rows)

        return result^

    fn padded_trailing(self, rows: Int, cols: Int) -> Self:
        _assert(rows >= 0 and cols >= 0, "Must specify rows >= 0 and cols >= 0 for Matrix.trailing_padded()")

        var src_data_ptr = self.unsafe_data_ptr()

        var new_rows = self._rows + rows
        var new_cols = self._cols + cols

        var result = Self(rows=new_rows, cols=new_cols)
        var result_data_ptr = result.unsafe_data_ptr()

        var row_offset = self._cols * self._scalar_depth()
        var new_row_offset = new_cols * self._scalar_depth()

        @parameter
        fn copy_row(row: Int):
            memcpy(dest=result_data_ptr.offset(row * new_row_offset), src=src_data_ptr.offset(row * row_offset), count=row_offset)

        parallelize[copy_row](self._rows)

        return result^

    fn horizontally_stacked(self, other: Self) -> Self:
        _assert(self._rows == other._rows, "Matrices must have same number of rows to stack horizontally")

        var result = Self(rows=self._rows, cols=self._cols + other._cols)

        var scalar_row_offset = self._cols * self._scalar_depth()
        var other_scalar_row_offset = other._cols * other._scalar_depth()
        var result_scalar_row_offset = result._cols * result._scalar_depth()
        var right_side_base_data_ptr = result.unsafe_data_ptr().offset(scalar_row_offset)

        @parameter
        fn process_row(row: Int):
            memcpy(
                dest=result.unsafe_data_ptr().offset(row * result_scalar_row_offset),
                src=self.unsafe_data_ptr().offset(row * scalar_row_offset),
                count=scalar_row_offset,
            )

            memcpy(
                dest=right_side_base_data_ptr.offset(row * result_scalar_row_offset),
                src=other.unsafe_data_ptr().offset(row * other_scalar_row_offset),
                count=other_scalar_row_offset,
            )

        parallelize[process_row](result._rows)

        return result^

    fn vertically_stacked(self, other: Self) -> Self:
        _assert(self._cols == other._cols, "Matrices must have same number of cols to stack vertically")

        var result = Self(rows=self._rows + other._rows, cols=self._cols)
        memcpy(dest=result.unsafe_data_ptr(), src=self.unsafe_data_ptr(), count=self._scalar_count())
        memcpy(dest=result.unsafe_data_ptr().offset(self._scalar_count()), src=other.unsafe_data_ptr(), count=other._scalar_count())

        return result^

    #
    # Type Conversion
    #
    fn as_type[new_dtype: DType](self) -> Matrix[new_dtype, depth, complex=complex]:
        @parameter
        if new_dtype == dtype:
            return rebind[UnsafePointer[Matrix[new_dtype, depth, complex=complex]]](UnsafePointer(to=self)).take_pointee()
        else:
            var result = Matrix[new_dtype, depth, complex=complex](rows=self._rows, cols=self._cols)

            @parameter
            fn convert_row(row: Int):
                @parameter
                fn convert_flattened_elements[width: Int](flattened_element: Int):
                    var index = self.flattened_index(row=row, offset=flattened_element)
                    var value = self._load[width](index).cast[new_dtype]()
                    result._store(value, index=index)

                vectorize[convert_flattened_elements, Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols * depth)

            parallelize[convert_row](self._rows)

            return result^

    fn as_complex[new_dtype: DType = dtype](self) -> Matrix[new_dtype, depth, complex=True]:
        @parameter
        if complex and new_dtype == dtype:
            return rebind[UnsafePointer[Matrix[new_dtype, depth, complex=True]]](UnsafePointer(to=self)).take_pointee()
        elif complex:
            var result = self.as_type[new_dtype]()

            return rebind[UnsafePointer[Matrix[new_dtype, depth, complex=True]]](UnsafePointer(to=result)).take_pointee()
        else:
            var result = Matrix[new_dtype, depth, complex=True](rows=self._rows, cols=self._cols)

            @parameter
            fn convert_row(row: Int):
                @parameter
                fn convert_flattened_elements[width: Int](flattened_element: Int):
                    var index = self.flattened_index(row=row, offset=flattened_element)
                    var value = self._load[width](index).as_complex[new_dtype]()
                    result._store(value, index=index)

                vectorize[convert_flattened_elements, Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols * depth)

            parallelize[convert_row](self._rows)

            return result^

    fn rebind[new_depth: Int](self) -> Matrix[dtype, new_depth, complex=complex]:
        constrained[new_depth == depth, "new_depth must be equal to depth for rebind"]()

        return rebind[UnsafePointer[Matrix[dtype, new_depth, complex=complex]]](UnsafePointer(to=self)).take_pointee()

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
                    writer.write(self._strided_load(row=row, col=col, component=component))

                    @parameter
                    if depth > 1:
                        writer.write(", " if component < depth - 1 else "]")

                writer.write(", " if col < self._cols - 1 else "")
            writer.write("],\n" if row < self._rows - 1 else "]\n")
        writer.write("  ]\n]")
