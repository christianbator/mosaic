#
# matrix.mojo
# mosaic
#
# Created by Christian Bator on 02/02/2025
#

from std.os import abort
from std.memory import unsafe_memset_zero, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.algorithm import vectorize
from max.algorithm import parallelize
from std.math import cos, sin, pi, floor, ceil, trunc, ceildiv, isclose, Ceilable, CeilDivable, Floorable, Truncable

from mosaic.utility import optimal_simd_width, unroll_factor, _assert

from .fft import fft, fft_dtype
from .number import Number, ScalarNumber
from .numeric_array import NumericArray
from .matrix_slice import MatrixSlice
from .strided_range import StridedRange


#
# Matrix
#
struct Matrix[dtype: DType, depth: Int = 1, *, complex: Bool = False](
    Absable, CeilDivable, Ceilable, Copyable, Equatable, Floorable, Movable, Roundable, Truncable, Writable
):
    #
    # Fields
    #
    comptime optimal_simd_width = optimal_simd_width[Self.dtype]() // 2 if Self.complex else optimal_simd_width[Self.dtype]()

    var _rows: Int
    var _cols: Int
    var _data: NumericArray[Self.dtype, complex=Self.complex]

    #
    # Initialization
    #
    def __init__(out self, *, rows: Int, cols: Int):
        comptime assert Self.depth > 0
        _assert(rows > 0 and cols > 0, "Rows and cols must be greather than 0")

        self._rows = rows
        self._cols = cols
        self._data = NumericArray[Self.dtype, complex=Self.complex](count=rows * cols * Self.depth)

    def __init__(out self, *, rows: Int, cols: Int, value: ScalarNumber[Self.dtype, complex=Self.complex]):
        comptime assert Self.depth > 0
        _assert(rows > 0 and cols > 0, "Rows and cols must be greather than 0")

        self._rows = rows
        self._cols = cols
        self._data = NumericArray[Self.dtype, complex=Self.complex](count=rows * cols * Self.depth)
        self.fill(value)

    def __init__(out self, *, rows: Int, cols: Int, var values: List[ScalarNumber[Self.dtype, complex=Self.complex]]):
        comptime assert Self.depth > 0
        _assert(rows * cols * Self.depth == len(values), "Mismatch in list length for Matrix constructor")

        self._rows = rows
        self._cols = cols
        self._data = NumericArray[Self.dtype, complex=Self.complex](
            values.unsafe_take_allocation().unsafe_leak(), count=rows * cols * Self.depth
        )

    def __init__(out self, *, rows: Int, cols: Int, var data: NumericArray[Self.dtype, complex=Self.complex]):
        comptime assert Self.depth > 0
        _assert(rows * cols * Self.depth == len(data), "Mismatch in data length for Matrix constructor")

        self._rows = rows
        self._cols = cols
        self._data = data^

    # This is an unsafe convenience constructor
    def __init__(out self, rows: Int, cols: Int, var data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]):
        comptime assert Self.depth > 0
        _assert(rows > 0 and cols > 0, "Rows and cols must be greather than 0")

        self._rows = rows
        self._cols = cols
        self._data = NumericArray[Self.dtype, complex=Self.complex](data, count=rows * cols * Self.depth)

    def __init__(out self, *, deinit move: Self):
        self._rows = move._rows
        self._cols = move._cols
        self._data = move._data^

    @staticmethod
    def ascending(*, rows: Int, cols: Int) -> Self:
        return Self(rows=rows, cols=cols, data=NumericArray[Self.dtype, complex=Self.complex].ascending(count=rows * cols * Self.depth))

    @staticmethod
    def random(
        *,
        rows: Int,
        cols: Int,
        min: Scalar[Self.dtype] = Scalar[Self.dtype].MIN_FINITE,
        max: Scalar[Self.dtype] = Scalar[Self.dtype].MAX_FINITE,
    ) -> Self:
        return Self(rows=rows, cols=cols, data=NumericArray[Self.dtype, complex=Self.complex].random(count=rows * cols * Self.depth))

    #
    # Properties
    #
    @always_inline
    def rows(self) -> Int:
        return self._rows

    @always_inline
    def cols(self) -> Int:
        return self._cols

    @always_inline
    def components(self) -> Int:
        return Self.depth

    @always_inline
    def count(self) -> Int:
        return self._rows * self._cols * Self.depth

    @always_inline
    def strided_count(self) -> Int:
        return self._rows * self._cols

    @always_inline
    def _scalar_count(self) -> Int:
        comptime if Self.complex:
            return 2 * self.count()
        else:
            return self.count()

    def _scalar_depth(self) -> Int:
        if Self.complex:
            return 2 * Self.depth
        else:
            return Self.depth

    #
    # Public Access
    #
    @always_inline
    def __getitem__(self, row: Int, col: Int) raises -> ScalarNumber[Self.dtype, complex=Self.complex] where Self.depth == 1:
        return self[row, col, 0]

    @always_inline
    def __getitem__(self, row: Int, col: Int, component: Int) raises -> ScalarNumber[Self.dtype, complex=Self.complex]:
        return self.strided_load(row=row, col=col, component=component)

    @always_inline
    def __setitem__(mut self, row: Int, col: Int, value: ScalarNumber[Self.dtype, complex=Self.complex]) raises where Self.depth == 1:
        self[row, col, 0] = value

    @always_inline
    def __setitem__(mut self, row: Int, col: Int, component: Int, value: ScalarNumber[Self.dtype, complex=Self.complex]) raises:
        self.strided_store(value, row=row, col=col, component=component)

    @always_inline
    def strided_load[
        width: Int = 1
    ](self, row: Int, col: Int) raises -> Number[Self.dtype, width, complex=Self.complex] where Self.depth == 1:
        return self.strided_load[width](row=row, col=col, component=0)

    @always_inline
    def strided_load[width: Int = 1](self, row: Int, col: Int, component: Int) raises -> Number[Self.dtype, width, complex=Self.complex]:
        return self._data.strided_load[width](index=self.index(row=row, col=col, component=component), stride=Self.depth)

    @always_inline
    def strided_store[
        width: Int, //
    ](mut self, value: Number[Self.dtype, width, complex=Self.complex], row: Int, col: Int,) raises where Self.depth == 1:
        self.strided_store(value, row=row, col=col, component=0)

    @always_inline
    def strided_store[
        width: Int, //
    ](mut self, value: Number[Self.dtype, width, complex=Self.complex], row: Int, col: Int, component: Int) raises:
        self._data.strided_store(value, index=self.index(row=row, col=col, component=component), stride=Self.depth)

    @always_inline
    def gather[
        width: Int, //
    ](
        self,
        row: Int,
        col: Int,
        component: Int,
        offset: SIMD[DType.int, width],
        mask: SIMD[DType.bool, width],
    ) -> Number[
        Self.dtype, width, complex=Self.complex
    ]:
        return self._data.gather(index=self.index(row=row, col=col, component=component), offset=offset, mask=mask)

    @always_inline
    def strided_gather[
        width: Int, //
    ](
        self,
        row: Int,
        col: Int,
        component: Int,
        offset: SIMD[DType.int, width],
        mask: SIMD[DType.bool, width],
    ) -> Number[
        Self.dtype, width, complex=Self.complex
    ]:
        return self.gather(row=row, col=col, component=component, offset=Self.depth * offset, mask=mask)

    @always_inline
    def scatter[
        width: Int, //
    ](
        self,
        value: Number[Self.dtype, width, complex=Self.complex],
        row: Int,
        col: Int,
        component: Int,
        offset: SIMD[DType.int, width],
        mask: SIMD[DType.bool, width],
    ):
        self._data.scatter(value, index=self.index(row=row, col=col, component=component), offset=offset, mask=mask)

    @always_inline
    def strided_scatter[
        width: Int, //
    ](
        self,
        row: Int,
        col: Int,
        component: Int,
        value: Number[Self.dtype, width, complex=Self.complex],
        offset: SIMD[DType.int, width],
        mask: SIMD[DType.bool, width],
    ):
        self.scatter(value, row=row, col=col, component=component, offset=Self.depth * offset, mask=mask)

    def load_full_depth(self, row: Int, col: Int) raises -> Array[ScalarNumber[Self.dtype, complex=Self.complex], Self.depth]:
        var result = Array[ScalarNumber[Self.dtype, complex=Self.complex], Self.depth](uninitialized=True)

        comptime for component in range(Self.depth):
            result[component] = self.strided_load(row=row, col=col, component=component)

        return result^

    def store_full_depth(mut self, value: Array[ScalarNumber[Self.dtype, complex=Self.complex], Self.depth], row: Int, col: Int) raises:
        comptime for component in range(Self.depth):
            self.strided_store(value[component], row=row, col=col, component=component)

    def create_full_depth_value(
        self, value: ScalarNumber[Self.dtype, complex=Self.complex]
    ) -> Array[ScalarNumber[Self.dtype, complex=Self.complex], Self.depth]:
        var result = Array[ScalarNumber[Self.dtype, complex=Self.complex], Self.depth](uninitialized=True)

        comptime for component in range(Self.depth):
            result[component] = value

        return result^

    def create_full_depth_value(
        self, *values: ScalarNumber[Self.dtype, complex=Self.complex]
    ) -> Array[ScalarNumber[Self.dtype, complex=Self.complex], Self.depth]:
        # TODO: Make this a compile-time check when possible
        debug_assert(Self.depth == len(values), "Mismatch in the number of values in the full depth value variadic constructor")

        var result = Array[ScalarNumber[Self.dtype, complex=Self.complex], Self.depth](uninitialized=True)

        comptime for component in range(Self.depth):
            result[component] = values[component]

        return result^

    def strided_iterate[
        HandlerType: def[width: Int](value: Number[Self.dtype, width, complex=Self.complex], row: Int, col: Int, component: Int) -> None, //
    ](self, handler: HandlerType):
        comptime for component in range(Self.depth):

            def visit_row(row: Int) {imm}:
                def visit_col[width: Int](col: Int) {imm}:
                    handler[width](
                        value=self._strided_load[width](row=row, col=col, component=component), row=row, col=col, component=component
                    )

                vectorize[
                    Self.optimal_simd_width,
                    unroll_factor=unroll_factor,
                ](self._cols, visit_col)

            parallelize(visit_row, self._rows)

    def strided_iterate_indices[HandlerType: def[width: Int](row: Int, col: Int, component: Int) -> None, //](self, handler: HandlerType):
        comptime for component in range(Self.depth):

            def visit_row(row: Int) {imm}:
                def visit_col[width: Int](col: Int) {imm}:
                    handler[width](row=row, col=col, component=component)

                vectorize[
                    Self.optimal_simd_width,
                    unroll_factor=unroll_factor,
                ](self._cols, visit_col)

            parallelize(visit_row, self._rows)

    def real(self) -> Matrix[Self.dtype, Self.depth, complex=False] where Self.complex:
        var result = Matrix[Self.dtype, Self.depth, complex=False](rows=self._rows, cols=self._cols)

        def take_real_value[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], row: Int, col: Int, component: Int) {mut result, imm}:
            result._strided_store(Number[Self.dtype, width, complex=False](value.real()), row=row, col=col, component=component)

        self.strided_iterate(take_real_value)

        return result^

    def imaginary(self) -> Matrix[Self.dtype, Self.depth, complex=False] where Self.complex:
        var result = Matrix[Self.dtype, Self.depth, complex=False](rows=self._rows, cols=self._cols)

        def take_imaginary_value[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], row: Int, col: Int, component: Int) {mut result, imm}:
            result._strided_store(Number[Self.dtype, width, complex=False](value.imaginary()), row=row, col=col, component=component)

        self.strided_iterate(take_imaginary_value)

        return result^

    #
    # Private Access
    #
    @always_inline
    def _strided_load[width: Int = 1](self, row: Int, col: Int) -> Number[Self.dtype, width, complex=Self.complex] where Self.depth == 1:
        return self._strided_load[width](row=row, col=col, component=0)

    @always_inline
    def _strided_load[width: Int = 1](self, row: Int, col: Int, component: Int) -> Number[Self.dtype, width, complex=Self.complex]:
        return self._data.strided_load[width](index=self.index(row=row, col=col, component=component), stride=Self.depth)

    @always_inline
    def _strided_store[
        width: Int, //
    ](mut self, value: Number[Self.dtype, width, complex=Self.complex], row: Int, col: Int,) where Self.depth == 1:
        self._strided_store(value, row=row, col=col, component=0)

    @always_inline
    def _strided_store[
        width: Int, //
    ](mut self, value: Number[Self.dtype, width, complex=Self.complex], row: Int, col: Int, component: Int):
        self._data.strided_store(value, index=self.index(row=row, col=col, component=component), stride=Self.depth)

    def _load_full_depth(self, row: Int, col: Int) -> Array[ScalarNumber[Self.dtype, complex=Self.complex], Self.depth]:
        var result = Array[ScalarNumber[Self.dtype, complex=Self.complex], Self.depth](uninitialized=True)

        comptime for component in range(Self.depth):
            result[component] = self._strided_load(row=row, col=col, component=component)

        return result^

    def _store_full_depth(mut self, value: Array[ScalarNumber[Self.dtype, complex=Self.complex], Self.depth], row: Int, col: Int):
        comptime for component in range(Self.depth):
            self._strided_store(value[component], row=row, col=col, component=component)

    @always_inline
    def _load[width: Int](self, index: Int) -> Number[Self.dtype, width, complex=Self.complex]:
        return self._data.load[width](index)

    @always_inline
    def _store[width: Int, //](mut self, value: Number[Self.dtype, width, complex=Self.complex], index: Int):
        self._data.store(value, index=index)

    #
    # Unsafe Access
    #
    @always_inline
    def unsafe_data_ptr(self) -> Pointer[Scalar[Self.dtype], MutUntrackedOrigin]:
        return self._data.unsafe_data_ptr()

    @always_inline
    def unsafe_uint8_ptr(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        return self._data.unsafe_uint8_ptr()

    def keep(self):
        pass

    #
    # Index Utilities
    #
    @always_inline
    def index(self, row: Int, col: Int, component: Int) -> Int:
        return (row * self._cols + col) * Self.depth + component

    @always_inline
    def flattened_index(self, row: Int, offset: Int) -> Int:
        return row * self._cols * Self.depth + offset

    @always_inline
    def row_end_index(self) -> Int:
        return self._rows - 1

    @always_inline
    def col_end_index(self) -> Int:
        return self._cols - 1

    def component_end_index(self) -> Int:
        return Self.depth - 1

    @always_inline
    def end_index(self) -> Int:
        return self.count() - 1

    #
    # Slicing
    #
    @always_inline
    def __getitem__[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self, row: Int, col_slice: Slice) raises -> MatrixSlice[
        StridedRange(Self.depth), Self.dtype, Self.depth, Self.complex, origin
    ]:
        return self[row : row + 1, col_slice]

    @always_inline
    def __getitem__[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self, row_slice: Slice, col: Int) raises -> MatrixSlice[
        StridedRange(Self.depth), Self.dtype, Self.depth, Self.complex, origin
    ]:
        return self[row_slice, col : col + 1]

    @always_inline
    def __getitem__[
        mut: Bool, origin: Origin[mut=mut], //
    ](ref[origin] self, row_slice: Slice, col_slice: Slice) raises -> MatrixSlice[
        StridedRange(Self.depth), Self.dtype, Self.depth, Self.complex, origin
    ]:
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
    def slice[
        mut: Bool, //, origin: Origin[mut=mut]
    ](ref[origin] self, row_range: StridedRange) raises -> MatrixSlice[
        StridedRange(Self.depth), Self.dtype, Self.depth, Self.complex, origin
    ]:
        return self.slice(row_range=row_range, col_range=StridedRange(self._cols))

    @always_inline
    def slice[
        mut: Bool, //, origin: Origin[mut=mut]
    ](ref[origin] self, *, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(Self.depth), Self.dtype, Self.depth, Self.complex, origin
    ]:
        return self.slice(row_range=StridedRange(self._rows), col_range=col_range)

    @always_inline
    def slice[
        mut: Bool, //, origin: Origin[mut=mut]
    ](ref[origin] self, row_range: StridedRange, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(Self.depth), Self.dtype, Self.depth, Self.complex, origin
    ]:
        return MatrixSlice[StridedRange(Self.depth), Self.dtype, Self.depth, Self.complex, origin](
            self, row_range=row_range, col_range=col_range
        )

    @always_inline
    def component_slice[
        mut: Bool, origin: Origin[mut=mut], //, component: Int
    ](ref[origin] self) -> MatrixSlice[StridedRange(component, component + 1), Self.dtype, Self.depth, Self.complex, origin]:
        return MatrixSlice[StridedRange(component, component + 1), Self.dtype, Self.depth, Self.complex, origin](self)

    @always_inline
    def component_slice[
        mut: Bool, origin: Origin[mut=mut], //, component: Int
    ](ref[origin] self, row_range: StridedRange) raises -> MatrixSlice[
        StridedRange(component, component + 1), Self.dtype, Self.depth, Self.complex, origin
    ]:
        return self.strided_slice[StridedRange(component, component + 1)](row_range)

    @always_inline
    def component_slice[
        mut: Bool, origin: Origin[mut=mut], //, component: Int
    ](ref[origin] self, *, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(component, component + 1), Self.dtype, Self.depth, Self.complex, origin
    ]:
        return self.strided_slice[StridedRange(component, component + 1)](col_range=col_range)

    @always_inline
    def component_slice[
        mut: Bool, origin: Origin[mut=mut], //, component: Int
    ](ref[origin] self, row_range: StridedRange, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(component, component + 1), Self.dtype, Self.depth, Self.complex, origin
    ]:
        return self.strided_slice[StridedRange(component, component + 1)](row_range=row_range, col_range=col_range)

    @always_inline
    def strided_slice[
        mut: Bool, origin: Origin[mut=mut], //, component_range: StridedRange
    ](ref[origin] self) -> MatrixSlice[component_range, Self.dtype, Self.depth, Self.complex, origin]:
        return MatrixSlice[component_range, Self.dtype, Self.depth, Self.complex, origin](self)

    @always_inline
    def strided_slice[
        mut: Bool, origin: Origin[mut=mut], //, component_range: StridedRange
    ](ref[origin] self, row_range: StridedRange) raises -> MatrixSlice[component_range, Self.dtype, Self.depth, Self.complex, origin]:
        return self.strided_slice[component_range](row_range=row_range, col_range=StridedRange(self._cols))

    @always_inline
    def strided_slice[
        mut: Bool, origin: Origin[mut=mut], //, component_range: StridedRange
    ](ref[origin] self, *, col_range: StridedRange) raises -> MatrixSlice[component_range, Self.dtype, Self.depth, Self.complex, origin]:
        return self.strided_slice[component_range](row_range=StridedRange(self._rows), col_range=col_range)

    @always_inline
    def strided_slice[
        mut: Bool, origin: Origin[mut=mut], //, component_range: StridedRange
    ](ref[origin] self, row_range: StridedRange, col_range: StridedRange) raises -> MatrixSlice[
        component_range, Self.dtype, Self.depth, Self.complex, origin
    ]:
        return MatrixSlice[component_range, Self.dtype, Self.depth, Self.complex, origin](self, row_range=row_range, col_range=col_range)

    def store_sub_matrix(mut self, value: Self, row: Int, col: Int, component: Int = 0) raises:
        self.store_sub_matrix(value[:, :], row=row, col=col, component=component)

    def store_sub_matrix[
        component_range: StridedRange, //
    ](mut self, matrix_slice: MatrixSlice[component_range, Self.dtype, _, Self.complex, _], row: Int, col: Int, component: Int = 0) raises:
        if (
            (component + component_range.count()) > Self.depth
            or (matrix_slice.row_range().end > self._rows)
            or (matrix_slice.col_range().end > self._cols)
        ):
            raise Error("Out of bounds sub-matrix store")

        comptime for matrix_slice_component in range(component_range.count()):

            def store_row(matrix_slice_row: Int) {mut self, imm}:
                def store_cols[
                    width: Int
                ](matrix_slice_col: Int) {mut self, imm matrix_slice, imm matrix_slice_row, imm row, imm col, imm component}:
                    self._strided_store(
                        matrix_slice._strided_load[width](row=matrix_slice_row, col=matrix_slice_col, component=matrix_slice_component),
                        row=row + matrix_slice_row,
                        col=col + matrix_slice_col,
                        component=component + matrix_slice_component,
                    )

                vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](matrix_slice.cols(), store_cols)

            parallelize(store_row, matrix_slice.rows())

    def strided_replication[new_depth: Int](self) -> Matrix[Self.dtype, new_depth, complex=Self.complex] where Self.depth == 1:
        comptime assert new_depth > Self.depth, "Strided replication requires a desired depth > 1"

        var result = Matrix[Self.dtype, new_depth, complex=Self.complex](rows=self._rows, cols=self._cols)

        def process_row(row: Int) {mut result, imm}:
            for col in range(self._cols):
                result._store_full_depth(result.create_full_depth_value(self._strided_load(row=row, col=col)), row=row, col=col)

        parallelize(process_row, self._rows)

        return result^

    @staticmethod
    def strided_replication(*, rows: Int, cols: Int, values: List[ScalarNumber[Self.dtype, complex=Self.complex]]) -> Self:
        _assert(rows * cols == len(values), "Mismatch in list length for Matrix strided replication constructor")

        var result = Self(rows=rows, cols=cols)

        def process_row(row: Int) {mut result, imm}:
            for col in range(cols):
                result._store_full_depth(result.create_full_depth_value(values[row * cols + col]), row=row, col=col)

        parallelize(process_row, rows)

        return result^

    def copied_to_component[
        component: Int, new_depth: Int
    ](self) -> Matrix[Self.dtype, new_depth, complex=Self.complex] where Self.depth == 1:
        comptime assert 0 <= component < new_depth, "Component must be within range of depth for copied_to_component()"

        var result = Matrix[Self.dtype, new_depth, complex=Self.complex](rows=self._rows, cols=self._cols)

        def process_row(row: Int) {mut result, imm}:
            def process_col[width: Int](col: Int) {mut result, imm self, imm row}:
                result._strided_store(self._strided_load[width](row=row, col=col), row=row, col=col, component=component)

            vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols, process_col)

        parallelize(process_row, self._rows)

        return result^

    #
    # Copyable
    #
    def __init__(out self, *, copy: Self):
        self = Self(rows=copy._rows, cols=copy._cols, data=copy._data.copy())

    def copy_into(self, mut other: Self):
        # This is bounds checked by NumericArray
        self._data.copy_into(other._data)

    #
    # Equatable
    #
    def __eq__(self, other: Self) -> Bool:
        if self.count() != other.count():
            return False

        for index in range(self.count()):
            comptime if Self.dtype.is_floating_point():
                if not isclose(self._data[index].value, other._data[index].value):
                    return False
            else:
                if self._data[index] != other._data[index]:
                    return False

        return True

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    def matches(self, other: Self, tolerance: Int = 1) -> Bool:
        if self.count() != other.count():
            return False

        for index in range(self.count()):
            comptime if Self.dtype.is_floating_point():
                if not isclose(
                    self._data[index].value,
                    other._data[index].value,
                    atol=Float64(tolerance),
                ):
                    return False
            else:
                var value = self._data[index].value.cast[DType.int64]()
                var other_value = other._data[index].value.cast[DType.int64]()
                if abs(value - other_value) > Scalar[DType.int64](tolerance):
                    return False

        return True

    #
    # Operators (Scalar)
    #
    def __neg__(self) -> Self:
        return self * -1

    def __add__(self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        var result = self.copy()
        result += rhs
        return result^

    def __iadd__(mut self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]):
        def add[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value + rhs

        self.for_each(add)

    def __sub__(self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        var result = self.copy()
        result -= rhs
        return result^

    def __isub__(mut self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]):
        def sub[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value - rhs

        self.for_each(sub)

    def __mul__(self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        var result = self.copy()
        result *= rhs
        return result^

    def __rmul__(self, lhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        return self * lhs

    def __imul__(mut self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]):
        def mul[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value * rhs

        self.for_each(mul)

    def __truediv__(self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        var result = self.copy()
        result /= rhs
        return result^

    def __itruediv__(mut self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]):
        def truediv[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value / rhs

        self.for_each(truediv)

    def __floordiv__(self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        var result = self.copy()
        result //= rhs
        return result^

    def __ifloordiv__(mut self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]):
        def floordiv[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value // rhs

        self.for_each(floordiv)

    def __mod__(self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        var result = self.copy()
        result %= rhs
        return result^

    def __imod__(mut self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]):
        def mod[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value % rhs

        self.for_each(mod)

    def __pow__(self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        var result = self.copy()
        result **= rhs
        return result^

    def __ipow__(mut self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]):
        def pow[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value**rhs

        self.for_each(pow)

    def __and__(self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        var result = self.copy()
        result &= rhs
        return result^

    def __iand__(mut self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]):
        def _and[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value & rhs

        self.for_each(_and)

    def __or__(self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        var result = self.copy()
        result |= rhs
        return result^

    def __ior__(mut self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]):
        def _or[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value | rhs

        self.for_each(_or)

    def __xor__(self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        var result = self.copy()
        result ^= rhs
        return result^

    def __ixor__(mut self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]):
        def _xor[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value ^ rhs

        self.for_each(_xor)

    def __lshift__(self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        var result = self.copy()
        result <<= rhs
        return result^

    def __ilshift__(mut self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]):
        def lshift[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value << rhs

        self.for_each(lshift)

    def __rshift__(self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]) -> Self:
        var result = self.copy()
        result >>= rhs
        return result^

    def __irshift__(mut self, rhs: ScalarNumber[Self.dtype, complex=Self.complex]):
        def rshift[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value >> rhs

        self.for_each(rshift)

    #
    # Operators (Matrix)
    #
    def __add__(self, rhs: Self) -> Self:
        var result = self.copy()
        result += rhs
        return result^

    def __iadd__(mut self, rhs: Self):
        def add[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return value + rhs

        self.for_each_zipped(add, rhs)

    def __sub__(self, rhs: Self) -> Self:
        var result = self.copy()
        result -= rhs
        return result^

    def __isub__(mut self, rhs: Self):
        def sub[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return value - rhs

        self.for_each_zipped(sub, rhs)

    def __mul__(self, rhs: Self) -> Self:
        var result = self.copy()
        result *= rhs
        return result^

    def __imul__(mut self, rhs: Self):
        def mul[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return value * rhs

        self.for_each_zipped(mul, rhs)

    def __matmul__(self, rhs: Self) -> Self:
        _assert(
            self._cols == rhs._rows,
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

        var shared_dim = self._cols

        comptime for component in range(Self.depth):

            def calculate_row(row: Int) {mut result, imm}:
                for k in range(shared_dim):

                    def dot_product[width: Int](col: Int) {mut result, imm self, imm rhs, imm row, imm k}:
                        result._strided_store(
                            row=row,
                            col=col,
                            component=component,
                            value=Number[Self.dtype, width, complex=Self.complex](
                                self._strided_load(row=row, col=k, component=component)
                            ).fma(
                                rhs._strided_load[width](row=k, col=col, component=component),
                                result._strided_load[width](row=row, col=col, component=component),
                            ),
                        )

                    vectorize[
                        Self.optimal_simd_width,
                        unroll_factor=unroll_factor,
                    ](result._cols, dot_product)

            parallelize(calculate_row, result._rows)

        return result^

    def __imatmul__(mut self, other: Self):
        _assert(
            self._rows == self._cols and self._rows == other._rows and self._cols == other._cols,
            "Dimension mismatch for in-place matrix multiplication",
        )

        (self @ other).copy_into(self)

    def __truediv__(self, rhs: Self) -> Self:
        var result = self.copy()
        result /= rhs
        return result^

    def __itruediv__(mut self, rhs: Self):
        def truediv[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return value / rhs

        self.for_each_zipped(truediv, rhs)

    def __floordiv__(self, rhs: Self) -> Self:
        var result = self.copy()
        result //= rhs
        return result^

    def __ifloordiv__(mut self, rhs: Self):
        def floordiv[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return value // rhs

        self.for_each_zipped(floordiv, rhs)

    def __mod__(self, rhs: Self) -> Self:
        var result = self.copy()
        result %= rhs
        return result^

    def __imod__(mut self, rhs: Self):
        def mod[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return value % rhs

        self.for_each_zipped(mod, rhs)

    def __pow__(self, rhs: Self) -> Self:
        var result = self.copy()
        result **= rhs
        return result^

    def __ipow__(mut self, rhs: Self):
        def pow[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return value**rhs

        self.for_each_zipped(pow, rhs)

    def __and__(self, rhs: Self) -> Self:
        var result = self.copy()
        result &= rhs
        return result^

    def __iand__(mut self, rhs: Self):
        def _and[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return value & rhs

        self.for_each_zipped(_and, rhs)

    def __or__(self, rhs: Self) -> Self:
        var result = self.copy()
        result |= rhs
        return result^

    def __ior__(mut self, rhs: Self):
        def _or[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return value | rhs

        self.for_each_zipped(_or, rhs)

    def __xor__(self, rhs: Self) -> Self:
        var result = self.copy()
        result ^= rhs
        return result^

    def __ixor__(mut self, rhs: Self):
        def _xor[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return value ^ rhs

        self.for_each_zipped(_xor, rhs)

    def __lshift__(self, rhs: Self) -> Self:
        var result = self.copy()
        result <<= rhs
        return result^

    def __ilshift__(mut self, rhs: Self):
        def lshift[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return value << rhs

        self.for_each_zipped(lshift, rhs)

    def __rshift__(self, rhs: Self) -> Self:
        var result = self.copy()
        result >>= rhs
        return result^

    def __irshift__(mut self, rhs: Self):
        def rshift[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return value >> rhs

        self.for_each_zipped(rshift, rhs)

    #
    # Numeric Traits
    #
    def __floor__(self) -> Self:
        var result = self.copy()

        def _floor[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return floor(value)

        result.for_each(_floor)
        return result^

    def __ceil__(self) -> Self:
        var result = self.copy()

        def _ceil[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return ceil(value)

        result.for_each(_ceil)
        return result^

    def __ceildiv__(self, rhs: Self) -> Self:
        var result = self.copy()

        def _ceildiv[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[
            Self.dtype, width, complex=Self.complex
        ]:
            return ceildiv(value, rhs)

        result.for_each_zipped(_ceildiv, rhs)

        return result^

    def __round__(self) -> Self:
        var result = self.copy()

        def _round[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return round(value)

        result.for_each(_round)
        return result^

    def __round__(self, ndigits: Int) -> Self:
        var result = self.copy()

        def _round[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return round(value, ndigits=ndigits)

        result.for_each(_round)
        return result^

    def __trunc__(self) -> Self:
        var result = self.copy()

        def _trunc[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return trunc(value)

        result.for_each(_trunc)
        return result^

    def __abs__(self) -> Self:
        var result = self.copy()

        def _abs[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return abs(value)

        result.for_each(_abs)
        return result^

    def norm(self) -> Matrix[Self.dtype, Self.depth, complex=False] where Self.complex:
        var result = Matrix[Self.dtype, Self.depth, complex=False](rows=self._rows, cols=self._cols)

        def transform_row(row: Int) {mut result, imm}:
            def transform_flattened_elements[width: Int](flattened_element: Int) {mut result, imm self, imm row}:
                var index = self.flattened_index(row=row, offset=flattened_element)
                result._store(self._load[width](index).norm(), index=index)

            vectorize[
                Self.optimal_simd_width,
                unroll_factor=unroll_factor,
            ](self._cols * Self.depth, transform_flattened_elements)

        parallelize(transform_row, self._rows)

        return result^

    #
    # Numeric Methods
    #
    def log(self) -> Self where Self.dtype.is_floating_point():
        var result = self.copy()

        def log[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value.log()

        result.for_each(log)
        return result^

    def clamp(mut self, lower_bound: ScalarNumber[Self.dtype], upper_bound: ScalarNumber[Self.dtype]):
        def transformer[
            width: Int
        ](value: Number[Self.dtype, width, complex=Self.complex]) {imm} -> Number[Self.dtype, width, complex=Self.complex]:
            return value.clamp(lower_bound=lower_bound, upper_bound=upper_bound)

        self.for_each(transformer)

    def clamped(self, lower_bound: ScalarNumber[Self.dtype], upper_bound: ScalarNumber[Self.dtype]) -> Self:
        var result = self.copy()
        result.clamp(lower_bound=lower_bound, upper_bound=upper_bound)

        return result^

    def strided_sum(self, component: Int) -> ScalarNumber[Self.dtype, complex=Self.complex]:
        var result = ScalarNumber[Self.dtype, complex=Self.complex](0)

        for row in range(self._rows):

            def process_cols[width: Int](col: Int) {mut result, imm self, imm row, imm component}:
                result += self._strided_load[width](row=row, col=col, component=component).reduce_add()

            vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols, process_cols)

        return result

    def strided_average(self, component: Int) -> ScalarNumber[DType.float64, complex=Self.complex]:
        return self.strided_sum(component).cast[DType.float64]() / Float64(self.strided_count())

    def strided_min(self, component: Int) -> ScalarNumber[Self.dtype, complex=Self.complex] where not Self.complex:
        var result = Scalar[Self.dtype].MAX_FINITE

        for row in range(self._rows):

            def process_cols[width: Int](col: Int) {mut result, imm self, imm row, imm component}:
                result = min(result, self._strided_load[width](row=row, col=col, component=component).reduce_min().value[0])

            vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols, process_cols)

        return result

    def strided_max(self, component: Int) -> ScalarNumber[Self.dtype, complex=Self.complex] where not Self.complex:
        var result = Scalar[Self.dtype].MIN_FINITE

        for row in range(self._rows):

            def process_cols[width: Int](col: Int) {mut result, imm self, imm row, imm component}:
                result = max(result, self._strided_load[width](row=row, col=col, component=component).reduce_max().value[0])

            vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols, process_cols)

        return result

    def strided_normalize(mut self):
        comptime for component in range(Self.depth):
            var strided_sum = self.strided_sum(component)

            def normalize[
                width: Int
            ](value: Number[Self.dtype, width, complex=Self.complex]) {var strided_sum, imm} -> Number[
                Self.dtype, width, complex=Self.complex
            ]:
                return value / strided_sum

            self.strided_for_each(normalize, component)

    def strided_fill(mut self, value: ScalarNumber[Self.dtype, complex=Self.complex], component: Int):
        def fill_row(row: Int) {mut self, imm}:
            def fill_cols[width: Int](col: Int) {mut self, imm value, imm row, imm component}:
                self._strided_store(Number[Self.dtype, width, complex=Self.complex](value), row=row, col=col, component=component)

            vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols, fill_cols)

        parallelize(fill_row, self._rows)

    def fill(mut self, value: ScalarNumber[Self.dtype, complex=Self.complex]):
        def fill_row(row: Int) {mut self, imm}:
            def fill_flattened_elements[width: Int](flattened_element: Int) {mut self, imm value, imm row}:
                self._store(
                    Number[Self.dtype, width, complex=Self.complex](value), index=self.flattened_index(row=row, offset=flattened_element)
                )

            vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols * Self.depth, fill_flattened_elements)

        parallelize(fill_row, self._rows)

    def strided_for_each[
        TransformerType: def[width: Int](value: Number[Self.dtype, width, complex=Self.complex]) -> Number[
            Self.dtype, width, complex=Self.complex
        ],
        //,
    ](mut self, transformer: TransformerType, component: Int):
        def transform_row(row: Int) {mut self, imm}:
            def transform_col[width: Int](col: Int) {mut self, imm transformer, imm row, imm component}:
                self._strided_store(
                    transformer[width](value=self._strided_load[width](row=row, col=col, component=component)),
                    row=row,
                    col=col,
                    component=component,
                )

            vectorize[
                Self.optimal_simd_width,
                unroll_factor=unroll_factor,
            ](self._cols, transform_col)

        parallelize(transform_row, self._rows)

    def strided_for_each_zipped[
        TransformerType: def[width: Int](
            value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]
        ) -> Number[Self.dtype, width, complex=Self.complex],
        //,
    ](mut self, transformer: TransformerType, other: Self, component: Int):
        _assert(
            self._rows == other._rows and self._cols == other._cols,
            "Cannot perform elementwise transformation on two matrices of different sizes",
        )

        def transform_row(row: Int) {mut self, imm}:
            def transform_cols[width: Int](col: Int) {mut self, imm transformer, imm other, imm row, imm component}:
                self._strided_store(
                    transformer[width](
                        value=self._strided_load[width](row=row, col=col, component=component),
                        rhs=other._strided_load[width](row=row, col=col, component=component),
                    ),
                    row=row,
                    col=col,
                    component=component,
                )

            vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols, transform_cols)

        parallelize(transform_row, self._rows)

    def for_each[
        TransformerType: def[width: Int](value: Number[Self.dtype, width, complex=Self.complex]) -> Number[
            Self.dtype, width, complex=Self.complex
        ],
        //,
    ](mut self, transformer: TransformerType):
        def transform_row(row: Int) {mut self, imm}:
            def transform_flattened_elements[width: Int](flattened_element: Int) {mut self, imm transformer, imm row}:
                var index = self.flattened_index(row=row, offset=flattened_element)
                self._store(transformer[width](self._load[width](index)), index=index)

            vectorize[
                Self.optimal_simd_width,
                unroll_factor=unroll_factor,
            ](self._cols * Self.depth, transform_flattened_elements)

        parallelize(transform_row, self._rows)

    def for_each_zipped[
        TransformerType: def[width: Int](
            value: Number[Self.dtype, width, complex=Self.complex], rhs: Number[Self.dtype, width, complex=Self.complex]
        ) -> Number[Self.dtype, width, complex=Self.complex],
        //,
    ](mut self, transformer: TransformerType, other: Self):
        _assert(
            self._rows == other._rows and self._cols == other._cols,
            "Cannot perform elementwise transformation on two matrices of different sizes",
        )

        def transform_row(row: Int) {mut self, imm}:
            def transform_flattened_elements[width: Int](flattened_element: Int) {mut self, imm transformer, imm other, imm row}:
                var index = self.flattened_index(row=row, offset=flattened_element)
                self._store(transformer[width](value=self._load[width](index), rhs=other._load[width](index)), index=index)

            vectorize[
                Self.optimal_simd_width,
                unroll_factor=unroll_factor,
            ](self._cols * Self.depth, transform_flattened_elements)

        parallelize(transform_row, self._rows)

    def map_to_range(
        mut self, min: ScalarNumber[Self.dtype, complex=Self.complex], max: ScalarNumber[Self.dtype, complex=Self.complex]
    ) where not Self.complex:
        comptime for component in range(Self.depth):
            var current_min = self.strided_min(component)
            var current_max = self.strided_max(component)
            var interpolation_factor = (max - min) / (current_max - current_min)

            def map[
                width: Int
            ](value: Number[Self.dtype, width, complex=Self.complex]) {var current_min, var interpolation_factor, imm} -> Number[
                Self.dtype, width, complex=Self.complex
            ]:
                return min + interpolation_factor * (value - current_min)

            self.strided_for_each(map, component)

    def mapped_to_range(
        self, min: ScalarNumber[Self.dtype, complex=Self.complex], max: ScalarNumber[Self.dtype, complex=Self.complex]
    ) -> Self where not Self.complex:
        var result = self.copy()
        result.map_to_range(min, max)

        return result^

    def fourier_transform[*, inverse: Bool = False](self) -> Matrix[fft_dtype, Self.depth, complex=True]:
        return fft[inverse=inverse](self)

    #
    # Geometric Transformations
    #
    def reshape(mut self, rows: Int, cols: Int):
        _assert(
            rows * cols == self.strided_count(),
            "Cannot reshape Matrix of strided_count = ",
            self.strided_count(),
            " to rows = ",
            rows,
            ", cols = ",
            cols,
        )

        self._rows = rows
        self._cols = cols

    def reshaped(self, rows: Int, cols: Int) -> Self:
        var result = self.copy()
        result.reshape(rows=rows, cols=cols)

        return result^

    def transpose(mut self):
        if self._rows == self._cols:
            # Iterate over upper triangle of square matrix and swap elements
            def iterate_over_row(row: Int) {mut self, imm}:
                for col in range(row + 1, self._cols):
                    var original_value = self._load_full_depth(row=row, col=col)
                    self._store_full_depth(self._load_full_depth(row=col, col=row), row=row, col=col)
                    self._store_full_depth(original_value, row=col, col=row)

            parallelize(iterate_over_row, self._rows)
        else:
            # Create a copy of the original
            var copy = self.copy()

            # Reshape self
            self.reshape(rows=self._cols, cols=self._rows)

            # Load from the copy and store in the transposed index of self
            def tranpose_row(row: Int) {mut self, imm}:
                for col in range(self._cols):
                    self._store_full_depth(copy._load_full_depth(row=col, col=row), row=row, col=col)

            parallelize(tranpose_row, self._rows)

    def transposed(self) -> Self:
        # Create a copy of the original
        var copy = self.copy()

        if self._rows == self._cols:
            copy.transpose()
        else:
            # Reshape the copy
            copy.reshape(rows=self._cols, cols=self._rows)

            # Load from self and store in the transposed index of copy
            def tranpose_row(row: Int) {mut copy, imm}:
                for col in range(copy._cols):
                    copy._store_full_depth(self._load_full_depth(row=col, col=row), row=row, col=col)

            parallelize(tranpose_row, copy._rows)

        return copy^

    def flip_horizontally(mut self):
        def process_row(row: Int) {mut self, imm}:
            for col in range(self._cols // 2):
                var original_value = self._load_full_depth(row=row, col=col)
                self._store_full_depth(self._load_full_depth(row=row, col=self.col_end_index() - col), row=row, col=col)
                self._store_full_depth(original_value, row=row, col=self.col_end_index() - col)

        parallelize(process_row, self._rows)

    def flipped_horizontally(self) -> Self:
        var result = self.copy()
        result.flip_horizontally()

        return result^

    def flip_vertically(mut self):
        def process_row(row: Int) {mut self, imm}:
            for col in range(self._cols):
                var original_value = self._load_full_depth(row=row, col=col)
                self._store_full_depth(self._load_full_depth(row=self.row_end_index() - row, col=col), row=row, col=col)
                self._store_full_depth(original_value, row=self.row_end_index() - row, col=col)

        parallelize(process_row, self._rows // 2)

    def flipped_vertically(self) -> Self:
        var result = self.copy()
        result.flip_vertically()

        return result^

    def rotate_90[*, clockwise: Bool](mut self):
        self.transpose()

        comptime if clockwise:
            self.flip_horizontally()
        else:
            self.flip_vertically()

    def rotated_90[*, clockwise: Bool](self) -> Self:
        var result = self.transposed()

        comptime if clockwise:
            result.flip_horizontally()
        else:
            result.flip_vertically()

        return result^

    def rotate_180(mut self):
        var row_range = ceildiv(self._rows, 2)

        def process_row(row: Int) {mut self, imm}:
            var col_range: Int
            if self._rows % 2 == 0:
                col_range = self._cols
            else:
                col_range = self._cols if row < row_range - 1 else self._cols // 2

            for col in range(col_range):
                var original_value = self._load_full_depth(row=row, col=col)
                self._store_full_depth(
                    self._load_full_depth(row=self.row_end_index() - row, col=self.col_end_index() - col), row=row, col=col
                )
                self._store_full_depth(original_value, row=self.row_end_index() - row, col=self.col_end_index() - col)

        parallelize(process_row, row_range)

    def rotated_180(self) -> Self:
        var result = self.copy()
        result.rotate_180()

        return result^

    def shift_origin_to_center(mut self):
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
            abort(String(error))

    def shifted_origin_to_center(self) -> Self:
        var result = self.copy()
        result.shift_origin_to_center()

        return result^

    def shift_center_to_origin(mut self):
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
            abort(String(error))

    def shifted_center_to_origin(self) -> Self:
        var result = self.copy()
        result.shift_center_to_origin()

        return result^

    def padded(self, size: Int) -> Self:
        return self.padded(rows=size, cols=size)

    def padded(self, rows: Int, cols: Int) -> Self:
        _assert(rows >= 0 and cols >= 0, "Must specify rows >= 0 and cols >= 0 for Matrix.padded()")

        var src_data_ptr = self.unsafe_data_ptr()

        var new_rows = self._rows + 2 * rows
        var new_cols = self._cols + 2 * cols

        var result = Self(rows=new_rows, cols=new_cols)
        var result_base_data_ptr = result.unsafe_data_ptr().unsafe_offset((rows * new_cols + cols) * self._scalar_depth())

        var row_offset = self._cols * self._scalar_depth()
        var new_row_offset = new_cols * self._scalar_depth()

        def copy_row(row: Int) {imm}:
            unsafe_memcpy(
                dest=result_base_data_ptr.unsafe_offset(row * new_row_offset),
                src=src_data_ptr.unsafe_offset(row * row_offset),
                count=row_offset,
            )

        parallelize(copy_row, self._rows)

        return result^

    def padded_trailing(self, rows: Int, cols: Int) -> Self:
        _assert(rows >= 0 and cols >= 0, "Must specify rows >= 0 and cols >= 0 for Matrix.trailing_padded()")

        var src_data_ptr = self.unsafe_data_ptr()

        var new_rows = self._rows + rows
        var new_cols = self._cols + cols

        var result = Self(rows=new_rows, cols=new_cols)
        var result_data_ptr = result.unsafe_data_ptr()

        var row_offset = self._cols * self._scalar_depth()
        var new_row_offset = new_cols * self._scalar_depth()

        def copy_row(row: Int) {imm}:
            unsafe_memcpy(
                dest=result_data_ptr.unsafe_offset(row * new_row_offset), src=src_data_ptr.unsafe_offset(row * row_offset), count=row_offset
            )

        parallelize(copy_row, self._rows)

        return result^

    def horizontally_stacked(self, other: Self) -> Self:
        _assert(self._rows == other._rows, "Matrices must have same number of rows to stack horizontally")

        var result = Self(rows=self._rows, cols=self._cols + other._cols)

        var scalar_row_offset = self._cols * self._scalar_depth()
        var other_scalar_row_offset = other._cols * other._scalar_depth()
        var result_scalar_row_offset = result._cols * result._scalar_depth()
        var right_side_base_data_ptr = result.unsafe_data_ptr().unsafe_offset(scalar_row_offset)

        def process_row(row: Int) {imm}:
            unsafe_memcpy(
                dest=result.unsafe_data_ptr().unsafe_offset(row * result_scalar_row_offset),
                src=self.unsafe_data_ptr().unsafe_offset(row * scalar_row_offset),
                count=scalar_row_offset,
            )

            unsafe_memcpy(
                dest=right_side_base_data_ptr.unsafe_offset(row * result_scalar_row_offset),
                src=other.unsafe_data_ptr().unsafe_offset(row * other_scalar_row_offset),
                count=other_scalar_row_offset,
            )

        parallelize(process_row, result._rows)

        return result^

    def vertically_stacked(self, other: Self) -> Self:
        _assert(self._cols == other._cols, "Matrices must have same number of cols to stack vertically")

        var result = Self(rows=self._rows + other._rows, cols=self._cols)
        unsafe_memcpy(dest=result.unsafe_data_ptr(), src=self.unsafe_data_ptr(), count=self._scalar_count())
        unsafe_memcpy(
            dest=result.unsafe_data_ptr().unsafe_offset(self._scalar_count()), src=other.unsafe_data_ptr(), count=other._scalar_count()
        )

        return result^

    #
    # Type Conversion
    #
    def as_type[new_dtype: DType](self) -> Matrix[new_dtype, Self.depth, complex=Self.complex]:
        comptime if new_dtype == Self.dtype:
            return rebind[Matrix[new_dtype, Self.depth, complex=Self.complex]](self).copy()
        else:
            var result = Matrix[new_dtype, Self.depth, complex=Self.complex](rows=self._rows, cols=self._cols)

            def convert_row(row: Int) {mut result, imm}:
                def convert_flattened_elements[width: Int](flattened_element: Int) {mut result, imm self, imm row}:
                    var index = self.flattened_index(row=row, offset=flattened_element)
                    var value = self._load[width](index).cast[new_dtype]()
                    result._store(value, index=index)

                vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols * Self.depth, convert_flattened_elements)

            parallelize(convert_row, self._rows)

            return result^

    def as_complex[new_dtype: DType = Self.dtype](self) -> Matrix[new_dtype, Self.depth, complex=True]:
        comptime if Self.complex and new_dtype == Self.dtype:
            return rebind[Matrix[new_dtype, Self.depth, complex=True]](self).copy()
        elif Self.complex:
            var result = self.as_type[new_dtype]()

            return rebind_var[Matrix[new_dtype, Self.depth, complex=True]](result^)
        else:
            var result = Matrix[new_dtype, Self.depth, complex=True](rows=self._rows, cols=self._cols)

            def convert_row(row: Int) {mut result, imm}:
                def convert_flattened_elements[width: Int](flattened_element: Int) {mut result, imm self, imm row}:
                    var index = self.flattened_index(row=row, offset=flattened_element)
                    var value = self._load[width](index).as_complex[new_dtype]()
                    result._store(value, index=index)

                vectorize[Self.optimal_simd_width, unroll_factor=unroll_factor](self._cols * Self.depth, convert_flattened_elements)

            parallelize(convert_row, self._rows)

            return result^

    def rebind[new_depth: Int](self) -> Matrix[Self.dtype, new_depth, complex=Self.complex]:
        comptime assert new_depth == Self.depth, "new_depth must be equal to depth for rebind"

        return rebind[Matrix[Self.dtype, new_depth, complex=Self.complex]](self).copy()

    #
    # Writable
    #
    def write_to[W: Writer](self, mut writer: W):
        writer.write("[Matrix:\n  [\n")

        for row in range(self._rows):
            writer.write("    [")
            for col in range(self._cols):
                comptime if Self.depth > 1:
                    writer.write("[")

                comptime for component in range(Self.depth):
                    writer.write(self._strided_load(row=row, col=col, component=component))

                    comptime if Self.depth > 1:
                        writer.write(", " if component < Self.depth - 1 else "]")

                writer.write(", " if col < self._cols - 1 else "")
            writer.write("],\n" if row < self._rows - 1 else "]\n")
        writer.write("  ]\n]")
