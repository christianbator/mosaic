#
# matrix_slice.mojo
# mosaic
#
# Created by Christian Bator on 03/15/2025
#

from std.memory import Pointer
from std.algorithm import vectorize
from max.algorithm import parallelize

from mosaic.utility import unroll_factor

from .number import Number, ScalarNumber
from .matrix import Matrix
from .strided_range import StridedRange


#
# MatrixSlice
#
struct MatrixSlice[
    mut: Bool,
    //,
    depth_range: StridedRange,
    dtype: DType,
    depth: Int,
    complex: Bool,
    origin: Origin[mut=mut],
](ImplicitlyCopyable, Movable, Writable):
    #
    # Fields
    #
    var _matrix: Pointer[Matrix[Self.dtype, Self.depth, complex=Self.complex], Self.origin]
    var _row_range: StridedRange
    var _col_range: StridedRange

    #
    # Initialization
    #
    def __init__(out self, ref[Self.origin] matrix: Matrix[Self.dtype, Self.depth, complex=Self.complex]):
        comptime assert Self.depth_range.end <= Self.depth, (
            "Out of bounds component range for matrix with depth " + String(Self.depth) + ": " + String(Self.depth_range)
        )

        self._matrix = Pointer(to=matrix)
        self._row_range = StridedRange(matrix.rows())
        self._col_range = StridedRange(matrix.cols())

    def __init__(
        out self,
        ref[Self.origin] matrix: Matrix[Self.dtype, Self.depth, complex=Self.complex],
        row_range: StridedRange,
        col_range: StridedRange,
    ) raises:
        comptime assert Self.depth_range.end <= Self.depth, (
            "Out of bounds component range for matrix with depth " + String(Self.depth) + ": " + String(Self.depth_range)
        )

        if row_range.end > matrix._rows or col_range.end > matrix._cols:
            raise Error(
                "Out of bounds matrix slice for matrix with size ",
                matrix._rows,
                " x ",
                matrix._cols,
                ": row_range: ",
                row_range,
                " col_range: ",
                col_range,
            )

        self._matrix = Pointer(to=matrix)
        self._row_range = row_range
        self._col_range = col_range

    def __init__[
        existing_depth_range: StridedRange
    ](out self, existing: MatrixSlice[existing_depth_range, Self.dtype, Self.depth, Self.complex, Self.origin]):
        comptime assert (existing_depth_range.start + Self.depth_range.end * existing_depth_range.step) <= Self.depth, (
            "Out of bounds component range for matrix with depth "
            + String(Self.depth)
            + ": "
            + String(existing_depth_range.start + Self.depth_range.end * existing_depth_range.step)
        )

        self._matrix = existing._matrix
        self._row_range = existing._row_range
        self._col_range = existing._col_range

    def __init__(out self, existing: Self, row_range: StridedRange, col_range: StridedRange) raises:
        comptime assert Self.depth_range.end <= Self.depth, (
            "Out of bounds component range for matrix with depth " + String(Self.depth) + ": " + String(Self.depth_range)
        )

        var new_row_range = StridedRange(
            existing._row_range.start + row_range.start * existing._row_range.step,
            existing._row_range.start + row_range.end * existing._row_range.step,
            existing._row_range.step * row_range.step,
        )

        var new_col_range = StridedRange(
            existing._col_range.start + col_range.start * existing._col_range.step,
            existing._col_range.start + col_range.end * existing._col_range.step,
            existing._col_range.step * col_range.step,
        )

        if new_row_range.end > existing._matrix[]._rows or new_col_range.end > existing._matrix[]._cols:
            raise Error(
                "Out of bounds matrix slice for matrix with size ",
                existing._matrix[]._rows,
                " x ",
                existing._matrix[]._cols,
                ": row_range: ",
                new_row_range,
                " col_range: ",
                new_col_range,
            )

        self._matrix = existing._matrix
        self._row_range = new_row_range
        self._col_range = new_col_range

    def __init__[
        existing_depth_range: StridedRange
    ](
        out self,
        existing: MatrixSlice[existing_depth_range, Self.dtype, Self.depth, Self.complex, Self.origin],
        row_range: StridedRange,
        col_range: StridedRange,
    ) raises:
        comptime assert (existing_depth_range.start + Self.depth_range.end * existing_depth_range.step) <= Self.depth, (
            "Out of bounds component range for matrix with depth "
            + String(Self.depth)
            + ": "
            + String(existing_depth_range.start + Self.depth_range.end * existing_depth_range.step)
        )

        var new_row_range = StridedRange(
            existing._row_range.start + row_range.start * existing._row_range.step,
            existing._row_range.start + row_range.end * existing._row_range.step,
            existing._row_range.step * row_range.step,
        )

        var new_col_range = StridedRange(
            existing._col_range.start + col_range.start * existing._col_range.step,
            existing._col_range.start + col_range.end * existing._col_range.step,
            existing._col_range.step * col_range.step,
        )

        if new_row_range.end > existing._matrix[]._rows or new_col_range.end > existing._matrix[]._cols:
            raise Error(
                "Out of bounds matrix slice for matrix with size ",
                existing._matrix[]._rows,
                " x ",
                existing._matrix[]._cols,
                ": row_range: ",
                new_row_range,
                " col_range: ",
                new_col_range,
            )

        self._matrix = existing._matrix
        self._row_range = new_row_range
        self._col_range = new_col_range

    #
    # Properties
    #
    @always_inline
    def row_range(self) -> StridedRange:
        return self._row_range

    @always_inline
    def col_range(self) -> StridedRange:
        return self._col_range

    def component_range(self) -> StridedRange:
        return Self.depth_range

    @always_inline
    def rows(self) -> Int:
        return self._row_range.count()

    @always_inline
    def cols(self) -> Int:
        return self._col_range.count()

    def components(self) -> Int:
        return Self.depth_range.count()

    #
    # Public Access
    #
    @always_inline
    def __getitem__(self, row: Int, col: Int) raises -> ScalarNumber[Self.dtype, complex=Self.complex]:
        comptime assert Self.depth_range.count() == 1, "Must specify component for matrix slice with depth > 1"

        return self[row, col, 0]

    @always_inline
    def __getitem__(self, row: Int, col: Int, component: Int) raises -> ScalarNumber[Self.dtype, complex=Self.complex]:
        return self.strided_load(row=row, col=col, component=component)

    @always_inline
    def __setitem__(
        mut self,
        row: Int,
        col: Int,
        value: ScalarNumber[Self.dtype, complex=Self.complex],
    ) raises where Self.mut:
        comptime assert Self.depth_range.count() == 1, "Must specify component for matrix slice with depth > 1"

        self[row, col, 0] = value

    @always_inline
    def __setitem__(
        mut self,
        row: Int,
        col: Int,
        component: Int,
        value: ScalarNumber[Self.dtype, complex=Self.complex],
    ) raises where Self.mut:
        self.strided_store(value, row=row, col=col, component=component)

    @always_inline
    def strided_load[width: Int = 1](self, row: Int, col: Int) raises -> Number[Self.dtype, width, complex=Self.complex]:
        comptime assert Self.depth_range.count() == 1, "Must specify component for matrix slice with depth > 1"

        return self.strided_load[width](row=row, col=col, component=0)

    @always_inline
    def strided_load[width: Int = 1](self, row: Int, col: Int, component: Int) raises -> Number[Self.dtype, width, complex=Self.complex]:
        return self._matrix[].strided_load[width](
            row=self._row_range.start + row * self._row_range.step,
            col=self._col_range.start + col * self._col_range.step,
            component=Self.depth_range.start + component * Self.depth_range.step,
        )

    @always_inline
    def strided_store[
        width: Int, //
    ](mut self, value: Number[Self.dtype, width, complex=Self.complex], row: Int, col: Int,) raises where Self.mut:
        comptime assert Self.depth_range.count() == 1, "Must specify component for matrix slice with depth > 1"

        self.strided_store(value, row=row, col=col, component=0)

    @always_inline
    def strided_store[
        width: Int, //
    ](mut self, value: Number[Self.dtype, width, complex=Self.complex], row: Int, col: Int, component: Int,) raises where Self.mut:
        self._matrix.unsafe_mut_cast[True]()[].strided_store(
            value,
            row=self._row_range.start + row * self._row_range.step,
            col=self._col_range.start + col * self._col_range.step,
            component=Self.depth_range.start + component * Self.depth_range.step,
        )

    #
    # Private Access
    #
    @always_inline
    def _strided_load[width: Int = 1](self, row: Int, col: Int) -> Number[Self.dtype, width, complex=Self.complex]:
        comptime assert Self.depth_range.count() == 1, "Must specify component for matrix slice with depth > 1"

        return self._strided_load[width](row=row, col=col, component=0)

    @always_inline
    def _strided_load[width: Int = 1](self, row: Int, col: Int, component: Int) -> Number[Self.dtype, width, complex=Self.complex]:
        return self._matrix[]._strided_load[width](
            row=self._row_range.start + row * self._row_range.step,
            col=self._col_range.start + col * self._col_range.step,
            component=Self.depth_range.start + component * Self.depth_range.step,
        )

    @always_inline
    def _strided_store[
        width: Int, //
    ](mut self, value: Number[Self.dtype, width, complex=Self.complex], row: Int, col: Int,) where Self.mut:
        comptime assert Self.depth_range.count() == 1, "Must specify component for matrix slice with depth > 1"

        self._strided_store(value, row=row, col=col, component=0)

    @always_inline
    def _strided_store[
        width: Int, //
    ](mut self, value: Number[Self.dtype, width, complex=Self.complex], row: Int, col: Int, component: Int,) where Self.mut:
        self._matrix.unsafe_mut_cast[True]()[]._strided_store(
            value,
            row=self._row_range.start + row * self._row_range.step,
            col=self._col_range.start + col * self._col_range.step,
            component=Self.depth_range.start + component * Self.depth_range.step,
        )

    #
    # Slicing
    #
    @always_inline
    def __getitem__(self, row: Int, col_slice: Slice) raises -> Self:
        return self[row : row + 1, col_slice]

    @always_inline
    def __getitem__(self, row_slice: Slice, col: Int) raises -> Self:
        return self[row_slice, col : col + 1]

    @always_inline
    def __getitem__(self, row_slice: Slice, col_slice: Slice) raises -> Self:
        return self.slice(
            row_range=StridedRange(
                slice=row_slice,
                default_start=0,
                default_end=self.rows(),
                default_step=1,
            ),
            col_range=StridedRange(
                slice=col_slice,
                default_start=0,
                default_end=self.cols(),
                default_step=1,
            ),
        )

    @always_inline
    def slice(self, row_range: StridedRange) raises -> Self:
        return self.slice(row_range=row_range, col_range=StridedRange(self.cols()))

    @always_inline
    def slice(self, *, col_range: StridedRange) raises -> Self:
        return self.slice(row_range=StridedRange(self.rows()), col_range=col_range)

    @always_inline
    def slice(self, row_range: StridedRange, col_range: StridedRange) raises -> Self:
        return Self(self, row_range=row_range, col_range=col_range)

    @always_inline
    def component_slice[
        component: Int
    ](self) -> MatrixSlice[
        StridedRange(
            Self.depth_range.start + component * Self.depth_range.step, Self.depth_range.start + component * Self.depth_range.step + 1
        ),
        Self.dtype,
        Self.depth,
        Self.complex,
        Self.origin,
    ]:
        return MatrixSlice[
            StridedRange(
                Self.depth_range.start + component * Self.depth_range.step, Self.depth_range.start + component * Self.depth_range.step + 1
            ),
            Self.dtype,
            Self.depth,
            Self.complex,
            Self.origin,
        ](self)

    @always_inline
    def component_slice[
        component: Int
    ](self, row_range: StridedRange) raises -> MatrixSlice[
        StridedRange(
            Self.depth_range.start + component * Self.depth_range.step, Self.depth_range.start + component * Self.depth_range.step + 1
        ),
        Self.dtype,
        Self.depth,
        Self.complex,
        Self.origin,
    ]:
        return self.component_slice[component](row_range=row_range, col_range=StridedRange(self.cols()))

    @always_inline
    def component_slice[
        component: Int
    ](self, *, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(
            Self.depth_range.start + component * Self.depth_range.step, Self.depth_range.start + component * Self.depth_range.step + 1
        ),
        Self.dtype,
        Self.depth,
        Self.complex,
        Self.origin,
    ]:
        return self.component_slice[component](row_range=StridedRange(self.rows()), col_range=col_range)

    @always_inline
    def component_slice[
        component: Int
    ](self, row_range: StridedRange, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(
            Self.depth_range.start + component * Self.depth_range.step, Self.depth_range.start + component * Self.depth_range.step + 1
        ),
        Self.dtype,
        Self.depth,
        Self.complex,
        Self.origin,
    ]:
        return MatrixSlice[
            StridedRange(
                Self.depth_range.start + component * Self.depth_range.step, Self.depth_range.start + component * Self.depth_range.step + 1
            ),
            Self.dtype,
            Self.depth,
            Self.complex,
            Self.origin,
        ](self, row_range=row_range, col_range=col_range)

    @always_inline
    def strided_slice[
        new_depth_range: StridedRange,
    ](self) -> MatrixSlice[
        StridedRange(
            Self.depth_range.start + new_depth_range.start * Self.depth_range.step,
            Self.depth_range.start + new_depth_range.end * Self.depth_range.step,
            Self.depth_range.step * new_depth_range.step,
        ),
        Self.dtype,
        Self.depth,
        Self.complex,
        Self.origin,
    ]:
        return MatrixSlice[
            StridedRange(
                Self.depth_range.start + new_depth_range.start * Self.depth_range.step,
                Self.depth_range.start + new_depth_range.end * Self.depth_range.step,
                Self.depth_range.step * new_depth_range.step,
            ),
            Self.dtype,
            Self.depth,
            Self.complex,
            Self.origin,
        ](self)

    @always_inline
    def strided_slice[
        new_depth_range: StridedRange,
    ](self, row_range: StridedRange) raises -> MatrixSlice[
        StridedRange(
            Self.depth_range.start + new_depth_range.start * Self.depth_range.step,
            Self.depth_range.start + new_depth_range.end * Self.depth_range.step,
            Self.depth_range.step * new_depth_range.step,
        ),
        Self.dtype,
        Self.depth,
        Self.complex,
        Self.origin,
    ]:
        return self.strided_slice[new_depth_range](row_range=row_range, col_range=StridedRange(self.cols()))

    @always_inline
    def strided_slice[
        new_depth_range: StridedRange,
    ](self, *, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(
            Self.depth_range.start + new_depth_range.start * Self.depth_range.step,
            Self.depth_range.start + new_depth_range.end * Self.depth_range.step,
            Self.depth_range.step * new_depth_range.step,
        ),
        Self.dtype,
        Self.depth,
        Self.complex,
        Self.origin,
    ]:
        return self.strided_slice[new_depth_range](row_range=StridedRange(self.rows()), col_range=col_range)

    @always_inline
    def strided_slice[
        new_depth_range: StridedRange,
    ](self, row_range: StridedRange, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(
            Self.depth_range.start + new_depth_range.start * Self.depth_range.step,
            Self.depth_range.start + new_depth_range.end * Self.depth_range.step,
            Self.depth_range.step * new_depth_range.step,
        ),
        Self.dtype,
        Self.depth,
        Self.complex,
        Self.origin,
    ]:
        return MatrixSlice[
            StridedRange(
                Self.depth_range.start + new_depth_range.start * Self.depth_range.step,
                Self.depth_range.start + new_depth_range.end * Self.depth_range.step,
                Self.depth_range.step * new_depth_range.step,
            ),
            Self.dtype,
            Self.depth,
            Self.complex,
            Self.origin,
        ](self, row_range=row_range, col_range=col_range)

    #
    # Deep Copy
    #
    def deep_copy[*, rebound_depth: Int = Self.depth_range.count()](self) -> Matrix[Self.dtype, rebound_depth, complex=Self.complex]:
        comptime assert rebound_depth == Self.depth_range.count(), "rebound_depth must equal matrix slice depth"

        var result = Matrix[Self.dtype, rebound_depth, complex=Self.complex](rows=self.rows(), cols=self.cols())

        comptime for slice_component in range(rebound_depth):
            var component = Self.depth_range.start + slice_component * Self.depth_range.step

            def process_row(range_row: Int) {mut result, imm}:
                var row = self._row_range.start + range_row * self._row_range.step

                def process_col[width: Int](range_col: Int) {mut result, imm self, imm row, imm component, imm range_row}:
                    var col = self._col_range.start + range_col * self._col_range.step

                    result._strided_store(
                        self._matrix[]._strided_load[width](row=row, col=col, component=component),
                        row=range_row,
                        col=range_col,
                        component=slice_component,
                    )

                vectorize[Matrix[Self.dtype, Self.depth, complex=Self.complex].optimal_simd_width, unroll_factor=unroll_factor](
                    self.cols(), process_col
                )

            parallelize(process_row, self.rows())

        return result^

    #
    # Numeric Methods
    #
    def fill(mut self, value: ScalarNumber[Self.dtype, complex=Self.complex]) where Self.mut:
        comptime for component in range(Self.depth_range.count()):

            def fill_row(row: Int) {mut self, imm}:
                def fill_cols[width: Int](col: Int) {mut self, imm value, imm row}:
                    self._strided_store(Number[Self.dtype, width, complex=Self.complex](value), row=row, col=col, component=component)

                vectorize[Matrix[Self.dtype, Self.depth, complex=Self.complex].optimal_simd_width, unroll_factor=unroll_factor](
                    self.cols(), fill_cols
                )

            parallelize(fill_row, self.rows())

    #
    # Writable
    #
    def write_to[W: Writer](self, mut writer: W):
        writer.write("[MatrixSlice:\n  [\n")

        for row in range(self.rows()):
            writer.write("    [")
            for col in range(self.cols()):
                comptime if Self.depth_range.count() > 1:
                    writer.write("[")

                comptime for component in range(Self.depth_range.count()):
                    writer.write(self._strided_load(row=row, col=col, component=component))

                    comptime if Self.depth_range.count() > 1:
                        writer.write(", " if component < Self.depth_range.count() - 1 else "]")

                writer.write(", " if col < self.cols() - 1 else "")
            writer.write("],\n" if row < self.rows() - 1 else "]\n")
        writer.write("  ]\n]")
