#
# test_matrix_numeric_methods.mojo
# mosaic
#
# Created by Christian Bator on 03/17/2025
#

from testing import assert_true

from mosaic.numeric import Matrix, Number, ScalarNumber


fn test_matrix_multiplication_int64_square() raises:
    var matrix = Matrix[DType.int64].ascending(rows=3, cols=3)

    var result = matrix @ matrix

    var correct_result = Matrix[DType.int64](rows=3, cols=3, values=List[ScalarNumber[DType.int64]](15, 18, 21, 42, 54, 66, 69, 90, 111))

    assert_true(result == correct_result)
