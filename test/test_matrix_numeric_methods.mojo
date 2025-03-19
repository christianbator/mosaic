#
# test_matrix_numeric_methods.mojo
# mosaic
#
# Created by Christian Bator on 03/17/2025
#

from testing import assert_equal

from mosaic.numeric import Matrix, Number, ScalarNumber

#
# TODO: Re-enable this test once stable Mojo catches up to nightly and this compiles
#
# fn test_matrix_multiplication_int32_square() raises:
#     var matrix = Matrix[DType.int32].ascending(rows=3, cols=3)

#     var result = matrix @ matrix

#     var correct_result = Matrix[DType.int32](rows=3, cols=3, values=List[ScalarNumber[DType.int32]](15, 18, 21, 42, 54, 66, 69, 90, 111))

#     assert_equal(result, correct_result)
