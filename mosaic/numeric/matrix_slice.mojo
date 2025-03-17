#
# matrix_slice.mojo
# mosaic
#
# Created by Christian Bator on 03/15/2025
#

from math import ceildiv
from memory import Pointer
from algorithm import parallelize, vectorize

from mosaic.utility import unroll_factor, fatal_error


#
# MatrixSlice
#
@value
struct MatrixSlice[
    mut: Bool, //,
    component_range: StridedRange,
    dtype: DType,
    depth: Int,
    complex: Bool,
    origin: Origin[mut],
](Stringable, Writable):
    #
    # Fields
    #
    alias _depth = ceildiv(component_range.end - component_range.start, component_range.step)

    var _matrix: Pointer[Matrix[dtype, depth, complex=complex], origin]
    var _row_range: StridedRange
    var _col_range: StridedRange
    var _rows: Int
    var _cols: Int

    #
    # Initialization
    #
    fn __init__(
        out self,
        ref [origin]matrix: Matrix[dtype, depth, complex=complex],
        row_range: StridedRange,
        col_range: StridedRange,
    ):
        self._matrix = Pointer.address_of(matrix)
        self._row_range = row_range
        self._col_range = col_range
        self._rows = ceildiv(row_range.end - row_range.start, row_range.step)
        self._cols = ceildiv(col_range.end - col_range.start, col_range.step)

    fn __init__[
        other_component_range: StridedRange
    ](out self, other: MatrixSlice[other_component_range, dtype, depth, complex, origin], row_range: StridedRange, col_range: StridedRange,):
        self._matrix = other._matrix

        self._row_range = StridedRange(
            other._row_range.start + row_range.start,
            other._row_range.start + row_range.end,
            other._row_range.step * row_range.step,
        )

        self._col_range = StridedRange(
            other._col_range.start + col_range.start,
            other._col_range.start + col_range.end,
            other._col_range.step * col_range.step,
        )

        self._rows = ceildiv(row_range.end - row_range.start, row_range.step)
        self._cols = ceildiv(col_range.end - col_range.start, col_range.step)

    #
    # Properties
    #
    @always_inline
    fn rows(self) -> Int:
        return self._rows

    @always_inline
    fn cols(self) -> Int:
        return self._cols

    @parameter
    fn components(self) -> Int:
        return Self._depth

    #
    # Access
    #
    @always_inline
    fn __getitem__(self, row: Int, col: Int) raises -> ScalarNumber[dtype, complex=complex]:
        constrained[Self._depth == 1, "Must specify component for matrix slice with depth > 1"]()

        return self[row, col, 0]

    @always_inline
    fn __getitem__(self, row: Int, col: Int, component: Int) raises -> ScalarNumber[dtype, complex=complex]:
        return self.strided_load[1](row=row, col=col, component=component)

    @always_inline
    fn __setitem__[
        origin: MutableOrigin, //
    ](mut self: MatrixSlice[_, dtype, _, complex, origin], row: Int, col: Int, value: ScalarNumber[dtype, complex=complex]) raises:
        constrained[Self._depth == 1, "Must specify component for matrix slice with depth > 1"]()

        self[row, col, 0] = value

    @always_inline
    fn __setitem__[
        origin: MutableOrigin, //
    ](mut self: MatrixSlice[_, dtype, _, complex, origin], row: Int, col: Int, component: Int, value: ScalarNumber[dtype, complex=complex]) raises:
        self.strided_store(value, row=row, col=col, component=component)

    @always_inline
    fn load[width: Int](self, row: Int, col: Int) raises -> Number[dtype, width, complex=complex]:
        constrained[Self._depth == 1, "Must specify component for matrix slice with depth > 1"]()

        return self.strided_load[width](row=row, col=col, component=0)

    @always_inline
    fn strided_load[width: Int](self, row: Int, col: Int, component: Int) raises -> Number[dtype, width, complex=complex]:
        return self._matrix[].strided_load[width](
            row=self._row_range.start + row * self._row_range.step,
            col=self._col_range.start + col * self._col_range.step,
            component=component_range.start + component * component_range.step,
        )

    @always_inline
    fn store[
        origin: MutableOrigin, width: Int, //
    ](mut self: MatrixSlice[_, dtype, _, complex, origin], value: Number[dtype, width, complex=complex], row: Int, col: Int) raises:
        constrained[Self._depth == 1, "Must specify component for matrix slice with depth > 1"]()

        self.strided_store(value, row=row, col=col, component=0)

    @always_inline
    fn strided_store[
        origin: MutableOrigin, width: Int, //
    ](mut self: MatrixSlice[_, dtype, _, complex, origin], value: Number[dtype, width, complex=complex], row: Int, col: Int, component: Int) raises:
        self._matrix[].strided_store(
            value,
            row=self._row_range.start + row * self._row_range.step,
            col=self._col_range.start + col * self._col_range.step,
            component=component_range.start + component * component_range.step,
        )

    #
    # Slicing
    #
    @always_inline
    fn __getitem__(self, row_slice: Slice, col_slice: Slice) -> Self:
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
    fn slice(self, row_range: StridedRange) -> Self:
        return self.slice(row_range=row_range, col_range=StridedRange(self._cols))

    @always_inline
    fn slice(self, *, col_range: StridedRange) -> Self:
        return self.slice(row_range=StridedRange(self._rows), col_range=col_range)

    @always_inline
    fn slice(self, row_range: StridedRange, col_range: StridedRange) -> Self:
        return Self(other=self, row_range=row_range, col_range=col_range)

    @always_inline
    fn component_slice[
        component: Int
    ](self) -> MatrixSlice[
        StridedRange(
            component_range.start + StridedRange(component, component + 1).start,
            component_range.start + StridedRange(component, component + 1).end,
            component_range.step * StridedRange(component, component + 1).step,
        ),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return self.strided_slice[StridedRange(component, component + 1)]()

    @always_inline
    fn component_slice[
        component: Int
    ](self, row_range: StridedRange) -> MatrixSlice[
        StridedRange(
            component_range.start + StridedRange(component, component + 1).start,
            component_range.start + StridedRange(component, component + 1).end,
            component_range.step * StridedRange(component, component + 1).step,
        ),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return self.strided_slice[StridedRange(component, component + 1)](row_range)

    @always_inline
    fn component_slice[
        component: Int
    ](self, *, col_range: StridedRange) -> MatrixSlice[
        StridedRange(
            component_range.start + StridedRange(component, component + 1).start,
            component_range.start + StridedRange(component, component + 1).end,
            component_range.step * StridedRange(component, component + 1).step,
        ),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return self.strided_slice[StridedRange(component, component + 1)](col_range=col_range)

    @always_inline
    fn component_slice[
        component: Int
    ](self, row_range: StridedRange, col_range: StridedRange) -> MatrixSlice[
        StridedRange(
            component_range.start + StridedRange(component, component + 1).start,
            component_range.start + StridedRange(component, component + 1).end,
            component_range.step * StridedRange(component, component + 1).step,
        ),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return self.strided_slice[StridedRange(component, component + 1)](row_range=row_range, col_range=col_range)

    @always_inline
    fn strided_slice[
        new_component_range: StridedRange,
    ](self) -> MatrixSlice[
        StridedRange(
            component_range.start + new_component_range.start,
            component_range.start + new_component_range.end,
            component_range.step * new_component_range.step,
        ),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return self.strided_slice[new_component_range](row_range=StridedRange(self._rows), col_range=StridedRange(self._cols))

    @always_inline
    fn strided_slice[
        new_component_range: StridedRange,
    ](self, row_range: StridedRange) -> MatrixSlice[
        StridedRange(
            component_range.start + new_component_range.start,
            component_range.start + new_component_range.end,
            component_range.step * new_component_range.step,
        ),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return self.strided_slice[new_component_range](row_range=row_range, col_range=StridedRange(self._cols))

    @always_inline
    fn strided_slice[
        new_component_range: StridedRange,
    ](self, *, col_range: StridedRange) -> MatrixSlice[
        StridedRange(
            component_range.start + new_component_range.start,
            component_range.start + new_component_range.end,
            component_range.step * new_component_range.step,
        ),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return self.strided_slice[new_component_range](row_range=StridedRange(self._rows), col_range=col_range)

    @always_inline
    fn strided_slice[
        new_component_range: StridedRange,
    ](self, row_range: StridedRange, col_range: StridedRange) -> MatrixSlice[
        StridedRange(
            component_range.start + new_component_range.start,
            component_range.start + new_component_range.end,
            component_range.step * new_component_range.step,
        ),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return MatrixSlice[
            StridedRange(
                component_range.start + new_component_range.start,
                component_range.start + new_component_range.end,
                component_range.step * new_component_range.step,
            ),
            dtype,
            depth,
            complex,
            origin,
        ](other=self, row_range=row_range, col_range=col_range)

    #
    # Copy
    #
    fn copy(self) -> Matrix[dtype, Self._depth, complex=complex]:
        var result = Matrix[dtype, Self._depth, complex=complex](rows=self._rows, cols=self._cols)

        @parameter
        for slice_component in range(Self._depth):
            var component = component_range.start + slice_component * component_range.step

            @parameter
            fn process_row(range_row: Int):
                var row = self._row_range.start + range_row * self._row_range.step

                @parameter
                fn process_col[width: Int](range_col: Int):
                    var col = self._col_range.start + range_col * self._col_range.step

                    try:
                        result.strided_store(
                            self._matrix[].strided_load[width](row=row, col=col, component=component),
                            row=range_row,
                            col=range_col,
                            component=slice_component,
                        )
                    except error:
                        fatal_error(error)

                vectorize[process_col, Matrix[dtype, depth, complex=complex].optimal_simd_width, unroll_factor=unroll_factor](self._cols)

            parallelize[process_row](self._rows)

        return result^

    fn rebound_copy[*, depth: Int](self) -> Matrix[dtype, depth, complex=complex]:
        constrained[depth == Self._depth]()

        var result = Matrix[dtype, depth, complex=complex](rows=self._rows, cols=self._cols)

        @parameter
        for slice_component in range(depth):
            var component = component_range.start + slice_component * component_range.step

            @parameter
            fn process_row(range_row: Int):
                var row = self._row_range.start + range_row * self._row_range.step

                @parameter
                fn process_col[width: Int](range_col: Int):
                    var col = self._col_range.start + range_col * self._col_range.step

                    try:
                        result.strided_store(
                            self._matrix[].strided_load[width](row=row, col=col, component=component), row=range_row, col=range_col, component=slice_component
                        )
                    except error:
                        fatal_error(error)

                vectorize[process_col, Matrix[dtype, depth, complex=complex].optimal_simd_width, unroll_factor=unroll_factor](self._cols)

            parallelize[process_row](self._rows)

        return result^

    #
    # Stringable & Writable
    #
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[MatrixSlice:\n  [\n")

        for row in range(self._rows):
            writer.write("    [")
            for col in range(self._cols):

                @parameter
                if Self._depth > 1:
                    writer.write("[")

                @parameter
                for component in range(Self._depth):
                    try:
                        writer.write(self[row, col, component])
                    except error:
                        fatal_error(error)

                    @parameter
                    if Self._depth > 1:
                        writer.write(", " if component < Self._depth - 1 else "]")

                writer.write(", " if col < self._cols - 1 else "")
            writer.write("],\n" if row < self._rows - 1 else "]\n")
        writer.write("  ]\n]")
