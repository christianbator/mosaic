#
# matrix_slice.mojo
# mosaic
#
# Created by Christian Bator on 03/15/2025
#

from memory import Pointer
from algorithm import parallelize, vectorize

from mosaic.utility import unroll_factor


#
# MatrixSlice
#
@value
struct MatrixSlice[
    mut: Bool, //,
    depth_range: StridedRange,
    dtype: DType,
    depth: Int,
    complex: Bool,
    origin: Origin[mut],
](Stringable, Writable):
    #
    # Fields
    #
    var _matrix: Pointer[Matrix[dtype, depth, complex=complex], origin]
    var _row_range: StridedRange
    var _col_range: StridedRange

    #
    # Initialization
    #
    fn __init__(out self, ref [origin]matrix: Matrix[dtype, depth, complex=complex]):
        constrained[
            depth_range.end <= depth,
            "Out of bounds component range for matrix with depth " + String(depth) + ": " + String(depth_range),
        ]()

        self._matrix = Pointer(to=matrix)
        self._row_range = StridedRange(matrix.rows())
        self._col_range = StridedRange(matrix.cols())

    fn __init__(out self, ref [origin]matrix: Matrix[dtype, depth, complex=complex], row_range: StridedRange, col_range: StridedRange) raises:
        constrained[
            depth_range.end <= depth,
            "Out of bounds component range for matrix with depth " + String(depth) + ": " + String(depth_range),
        ]()

        if row_range.end > matrix._rows or col_range.end > matrix._cols:
            raise Error(
                "Out of bounds matrix slice for matrix with size ", matrix._rows, " x ", matrix._cols, ": row_range: ", row_range, " col_range: ", col_range
            )

        self._matrix = Pointer(to=matrix)
        self._row_range = row_range
        self._col_range = col_range

    fn __init__[existing_depth_range: StridedRange](out self, existing: MatrixSlice[existing_depth_range, dtype, depth, complex, origin]):
        constrained[
            (existing_depth_range.start + depth_range.end * existing_depth_range.step) <= depth,
            "Out of bounds component range for matrix with depth "
            + String(depth)
            + ": "
            + String(existing_depth_range.start + depth_range.end * existing_depth_range.step),
        ]()

        self._matrix = existing._matrix
        self._row_range = existing._row_range
        self._col_range = existing._col_range

    fn __init__(out self, existing: Self, row_range: StridedRange, col_range: StridedRange) raises:
        constrained[
            depth_range.end <= depth,
            "Out of bounds component range for matrix with depth " + String(depth) + ": " + String(depth_range),
        ]()

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

    fn __init__[
        existing_depth_range: StridedRange
    ](out self, existing: MatrixSlice[existing_depth_range, dtype, depth, complex, origin], row_range: StridedRange, col_range: StridedRange) raises:
        constrained[
            (existing_depth_range.start + depth_range.end * existing_depth_range.step) <= depth,
            "Out of bounds component range for matrix with depth "
            + String(depth)
            + ": "
            + String(existing_depth_range.start + depth_range.end * existing_depth_range.step),
        ]()

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
    fn row_range(self) -> StridedRange:
        return self._row_range

    @always_inline
    fn col_range(self) -> StridedRange:
        return self._col_range

    @parameter
    fn component_range(self) -> StridedRange:
        return depth_range

    @always_inline
    fn rows(self) -> Int:
        return self._row_range.count()

    @always_inline
    fn cols(self) -> Int:
        return self._col_range.count()

    @parameter
    fn components(self) -> Int:
        return depth_range.count()

    #
    # Public Access
    #
    @always_inline
    fn __getitem__(self, row: Int, col: Int) raises -> ScalarNumber[dtype, complex=complex]:
        constrained[depth_range.count() == 1, "Must specify component for matrix slice with depth > 1"]()

        return self[row, col, 0]

    @always_inline
    fn __getitem__(self, row: Int, col: Int, component: Int) raises -> ScalarNumber[dtype, complex=complex]:
        return self.strided_load(row=row, col=col, component=component)

    @always_inline
    fn __setitem__[
        origin: MutableOrigin, //
    ](mut self: MatrixSlice[_, dtype, _, complex, origin], row: Int, col: Int, value: ScalarNumber[dtype, complex=complex]) raises:
        constrained[depth_range.count() == 1, "Must specify component for matrix slice with depth > 1"]()

        self[row, col, 0] = value

    @always_inline
    fn __setitem__[
        origin: MutableOrigin, //
    ](mut self: MatrixSlice[_, dtype, _, complex, origin], row: Int, col: Int, component: Int, value: ScalarNumber[dtype, complex=complex]) raises:
        self.strided_store(value, row=row, col=col, component=component)

    @always_inline
    fn strided_load[width: Int = 1](self, row: Int, col: Int) raises -> Number[dtype, width, complex=complex]:
        constrained[depth_range.count() == 1, "Must specify component for matrix slice with depth > 1"]()

        return self.strided_load[width](row=row, col=col, component=0)

    @always_inline
    fn strided_load[width: Int = 1](self, row: Int, col: Int, component: Int) raises -> Number[dtype, width, complex=complex]:
        return self._matrix[].strided_load[width](
            row=self._row_range.start + row * self._row_range.step,
            col=self._col_range.start + col * self._col_range.step,
            component=depth_range.start + component * depth_range.step,
        )

    @always_inline
    fn strided_store[
        origin: MutableOrigin, width: Int, //
    ](mut self: MatrixSlice[_, dtype, _, complex, origin], value: Number[dtype, width, complex=complex], row: Int, col: Int) raises:
        constrained[depth_range.count() == 1, "Must specify component for matrix slice with depth > 1"]()

        self.strided_store(value, row=row, col=col, component=0)

    @always_inline
    fn strided_store[
        origin: MutableOrigin, width: Int, //
    ](mut self: MatrixSlice[_, dtype, _, complex, origin], value: Number[dtype, width, complex=complex], row: Int, col: Int, component: Int) raises:
        self._matrix[].strided_store(
            value,
            row=self._row_range.start + row * self._row_range.step,
            col=self._col_range.start + col * self._col_range.step,
            component=depth_range.start + component * depth_range.step,
        )

    #
    # Private Access
    #
    @always_inline
    fn _strided_load[width: Int = 1](self, row: Int, col: Int) -> Number[dtype, width, complex=complex]:
        constrained[depth_range.count() == 1, "Must specify component for matrix slice with depth > 1"]()

        return self._strided_load[width](row=row, col=col, component=0)

    @always_inline
    fn _strided_load[width: Int = 1](self, row: Int, col: Int, component: Int) -> Number[dtype, width, complex=complex]:
        return self._matrix[]._strided_load[width](
            row=self._row_range.start + row * self._row_range.step,
            col=self._col_range.start + col * self._col_range.step,
            component=depth_range.start + component * depth_range.step,
        )

    @always_inline
    fn _strided_store[
        origin: MutableOrigin, width: Int, //
    ](mut self: MatrixSlice[_, dtype, _, complex, origin], value: Number[dtype, width, complex=complex], row: Int, col: Int):
        constrained[depth_range.count() == 1, "Must specify component for matrix slice with depth > 1"]()

        self._strided_store(value, row=row, col=col, component=0)

    @always_inline
    fn _strided_store[
        origin: MutableOrigin, width: Int, //
    ](mut self: MatrixSlice[_, dtype, _, complex, origin], value: Number[dtype, width, complex=complex], row: Int, col: Int, component: Int):
        self._matrix[]._strided_store(
            value,
            row=self._row_range.start + row * self._row_range.step,
            col=self._col_range.start + col * self._col_range.step,
            component=depth_range.start + component * depth_range.step,
        )

    #
    # Slicing
    #
    @always_inline
    fn __getitem__(self, row: Int, col_slice: Slice) raises -> Self:
        return self[row : row + 1, col_slice]

    @always_inline
    fn __getitem__(self, row_slice: Slice, col: Int) raises -> Self:
        return self[row_slice, col : col + 1]

    @always_inline
    fn __getitem__(self, row_slice: Slice, col_slice: Slice) raises -> Self:
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
    fn slice(self, row_range: StridedRange) raises -> Self:
        return self.slice(row_range=row_range, col_range=StridedRange(self.cols()))

    @always_inline
    fn slice(self, *, col_range: StridedRange) raises -> Self:
        return self.slice(row_range=StridedRange(self.rows()), col_range=col_range)

    @always_inline
    fn slice(self, row_range: StridedRange, col_range: StridedRange) raises -> Self:
        return Self(self, row_range=row_range, col_range=col_range)

    @always_inline
    fn component_slice[
        component: Int
    ](self) -> MatrixSlice[
        StridedRange(depth_range.start + component * depth_range.step, depth_range.start + component * depth_range.step + 1),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return MatrixSlice[
            StridedRange(depth_range.start + component * depth_range.step, depth_range.start + component * depth_range.step + 1),
            dtype,
            depth,
            complex,
            origin,
        ](self)

    @always_inline
    fn component_slice[
        component: Int
    ](self, row_range: StridedRange) raises -> MatrixSlice[
        StridedRange(depth_range.start + component * depth_range.step, depth_range.start + component * depth_range.step + 1),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return self.component_slice[component](row_range=row_range, col_range=StridedRange(self.cols()))

    @always_inline
    fn component_slice[
        component: Int
    ](self, *, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(depth_range.start + component * depth_range.step, depth_range.start + component * depth_range.step + 1),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return self.component_slice[component](row_range=StridedRange(self.rows()), col_range=col_range)

    @always_inline
    fn component_slice[
        component: Int
    ](self, row_range: StridedRange, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(depth_range.start + component * depth_range.step, depth_range.start + component * depth_range.step + 1),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return MatrixSlice[
            StridedRange(depth_range.start + component * depth_range.step, depth_range.start + component * depth_range.step + 1),
            dtype,
            depth,
            complex,
            origin,
        ](self, row_range=row_range, col_range=col_range)

    @always_inline
    fn strided_slice[
        new_depth_range: StridedRange,
    ](self) -> MatrixSlice[
        StridedRange(
            depth_range.start + new_depth_range.start * depth_range.step,
            depth_range.start + new_depth_range.end * depth_range.step,
            depth_range.step * new_depth_range.step,
        ),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return MatrixSlice[
            StridedRange(
                depth_range.start + new_depth_range.start * depth_range.step,
                depth_range.start + new_depth_range.end * depth_range.step,
                depth_range.step * new_depth_range.step,
            ),
            dtype,
            depth,
            complex,
            origin,
        ](self)

    @always_inline
    fn strided_slice[
        new_depth_range: StridedRange,
    ](self, row_range: StridedRange) raises -> MatrixSlice[
        StridedRange(
            depth_range.start + new_depth_range.start * depth_range.step,
            depth_range.start + new_depth_range.end * depth_range.step,
            depth_range.step * new_depth_range.step,
        ),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return self.strided_slice[new_depth_range](row_range=row_range, col_range=StridedRange(self.cols()))

    @always_inline
    fn strided_slice[
        new_depth_range: StridedRange,
    ](self, *, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(
            depth_range.start + new_depth_range.start * depth_range.step,
            depth_range.start + new_depth_range.end * depth_range.step,
            depth_range.step * new_depth_range.step,
        ),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return self.strided_slice[new_depth_range](row_range=StridedRange(self.rows()), col_range=col_range)

    @always_inline
    fn strided_slice[
        new_depth_range: StridedRange,
    ](self, row_range: StridedRange, col_range: StridedRange) raises -> MatrixSlice[
        StridedRange(
            depth_range.start + new_depth_range.start * depth_range.step,
            depth_range.start + new_depth_range.end * depth_range.step,
            depth_range.step * new_depth_range.step,
        ),
        dtype,
        depth,
        complex,
        origin,
    ]:
        return MatrixSlice[
            StridedRange(
                depth_range.start + new_depth_range.start * depth_range.step,
                depth_range.start + new_depth_range.end * depth_range.step,
                depth_range.step * new_depth_range.step,
            ),
            dtype,
            depth,
            complex,
            origin,
        ](self, row_range=row_range, col_range=col_range)

    #
    # Copy
    #
    fn copy[*, rebound_depth: Int = depth_range.count()](self) -> Matrix[dtype, rebound_depth, complex=complex]:
        constrained[rebound_depth == depth_range.count(), "rebound_depth must equal matrix slice depth"]()

        var result = Matrix[dtype, rebound_depth, complex=complex](rows=self.rows(), cols=self.cols())

        @parameter
        for slice_component in range(rebound_depth):
            var component = depth_range.start + slice_component * depth_range.step

            @parameter
            fn process_row(range_row: Int):
                var row = self._row_range.start + range_row * self._row_range.step

                @parameter
                fn process_col[width: Int](range_col: Int):
                    var col = self._col_range.start + range_col * self._col_range.step

                    result._strided_store(
                        self._matrix[]._strided_load[width](row=row, col=col, component=component),
                        row=range_row,
                        col=range_col,
                        component=slice_component,
                    )

                vectorize[process_col, Matrix[dtype, depth, complex=complex].optimal_simd_width, unroll_factor=unroll_factor](self.cols())

            parallelize[process_row](self.rows())

        return result^

    #
    # Numeric Methods
    #
    fn fill[origin: MutableOrigin, //](mut self: MatrixSlice[_, dtype, _, complex, origin], value: ScalarNumber[dtype, complex=complex]):
        @parameter
        for component in range(depth_range.count()):

            @parameter
            fn fill_row(row: Int):
                @parameter
                fn fill_cols[width: Int](col: Int):
                    self._strided_store(Number[dtype, width, complex=complex](value), row=row, col=col, component=component)

                vectorize[fill_cols, Matrix[dtype, depth, complex=complex].optimal_simd_width, unroll_factor=unroll_factor](self.cols())

            parallelize[fill_row](self.rows())

    #
    # Stringable & Writable
    #
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[MatrixSlice:\n  [\n")

        for row in range(self.rows()):
            writer.write("    [")
            for col in range(self.cols()):

                @parameter
                if depth_range.count() > 1:
                    writer.write("[")

                @parameter
                for component in range(depth_range.count()):
                    writer.write(self._strided_load(row=row, col=col, component=component))

                    @parameter
                    if depth_range.count() > 1:
                        writer.write(", " if component < depth_range.count() - 1 else "]")

                writer.write(", " if col < self.cols() - 1 else "")
            writer.write("],\n" if row < self.rows() - 1 else "]\n")
        writer.write("  ]\n]")
