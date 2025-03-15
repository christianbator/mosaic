#
# matrix_slice.mojo
# mosaic
#
# Created by Christian Bator on 03/15/2025
#

from math import ceildiv
from memory import Pointer
from algorithm import parallelize, vectorize

#
# MatrixSlice
#
@value
struct MatrixSlice[
    mut: Bool, //,
    component_slice: Slice,
    dtype: DType,
    depth: Int,
    complex: Bool,
    origin: Origin[mut]
](Stringable, Writable):
    alias _MatrixType = Matrix[dtype, depth, complex]

    var _matrix: Pointer[Self._MatrixType, origin]

    var _row_start: Int
    var _row_end: Int
    var _row_step: Int
    var _rows: Int

    var _col_start: Int
    var _col_end: Int
    var _col_step: Int
    var _cols: Int

    alias _component_start = component_slice.start.value() if component_slice.start else 0
    alias _component_end = component_slice.end.value() if component_slice.end else depth
    alias _component_step = component_slice.step.value() if component_slice.step else 1
    alias _depth = ceildiv(Self._component_end - Self._component_start, Self._component_step)

    fn __init__(out self, ref [origin] matrix: Self._MatrixType, row_slice: Slice, col_slice: Slice):
        self._matrix = Pointer.address_of(matrix)
        self._row_start = row_slice.start.value() if row_slice.start else 0
        self._row_end = row_slice.end.value() if row_slice.end else matrix.rows()
        self._row_step = row_slice.step.value() if row_slice.step else 1
        self._rows = ceildiv(self._row_end - self._row_start, self._row_step)

        self._col_start = col_slice.start.value() if col_slice.start else 0
        self._col_end = col_slice.end.value() if col_slice.end else matrix.cols()
        self._col_step = col_slice.step.value() if col_slice.step else 1
        self._cols = ceildiv(self._col_end - self._col_start, self._col_step)

    @always_inline
    fn rows(self) -> Int:
        return self._rows
    
    @always_inline
    fn cols(self) -> Int:
        return self._cols
    
    @always_inline
    fn components(self) -> Int:
        return Self._depth

    #
    # Access
    #
    fn __getitem__(self, row: Int, col: Int) -> ScalarNumber[dtype, complex]:
        constrained[Self._depth == 1, "Must specify component for matrix slice with depth > 1"]()

        return self.strided_load[1](row = row, col = col, component = 0)
    
    fn __getitem__(self, row: Int, col: Int, component: Int) -> ScalarNumber[dtype, complex]:
        return self.strided_load[1](row = row, col = col, component = component)

    fn __setitem__[origin: MutableOrigin, //](mut self: MatrixSlice[_, dtype, _, complex, origin], row: Int, col: Int, value: ScalarNumber[dtype, complex]):
        constrained[Self._depth == 1, "Must specify component for matrix slice with depth > 1"]()

        self.strided_store(row = row, col = col, component = 0, value = value)

    fn __setitem__[origin: MutableOrigin, //](mut self: MatrixSlice[_, dtype, _, complex, origin], row: Int, col: Int, component: Int, value: ScalarNumber[dtype, complex]):
        self.strided_store(row = row, col = col, component = component, value = value)

    fn strided_load[width: Int](self, row: Int, col: Int, component: Int) -> Number[dtype, complex, width]:
        return self._matrix[].strided_load[width](
            row = self._row_start + row * self._row_step,
            col = self._col_start + col * self._col_step,
            component = Self._component_start + component * Self._component_step
        )

    fn strided_store[origin: MutableOrigin, width: Int, //](mut self: MatrixSlice[_, dtype, _, complex, origin], row: Int, col: Int, component: Int, value: Number[dtype, complex, width]):
        self._matrix[].strided_store(
            row = self._row_start + row * self._row_step,
            col = self._col_start + col * self._col_step,
            component = Self._component_start + component * Self._component_step,
            value = value
        )

    #
    # Copy
    #
    fn copy(self) -> Matrix[dtype, Self._depth, complex]:
        var result = Matrix[dtype, Self._depth, complex](rows = self.rows(), cols = self.cols())

        @parameter
        for slice_component in range(Self._depth):
            var component = Self._component_start + slice_component * Self._component_step

            @parameter
            fn process_row(slice_row: Int):
                var row = self._row_start + slice_row * self._row_step
            
                @parameter
                fn process_col[width: Int](slice_col: Int):
                    var col = self._col_start + slice_col * self._col_step

                    result.strided_store(
                        row = slice_row,
                        col = slice_col,
                        component = slice_component,
                        value = self._matrix[].strided_load[width](row = row, col = col, component = component)
                    )

                vectorize[process_col, Self._MatrixType.optimal_simd_width](self._cols)
            parallelize[process_row](self._rows)

        return result^
    
    #
    # Numeric Methods
    #


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
                for component in range(Self._depth):
                    writer.write(self[row, col, component])
                writer.write(", " if col < self._cols - 1 else "")
            writer.write("],\n" if row < self._rows - 1 else "]\n")
        writer.write("  ]\n]")
