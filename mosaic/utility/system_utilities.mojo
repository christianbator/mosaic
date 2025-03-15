#
# system_utilities.mojo
# mosaic
#
# Created by Christian Bator on 02/02/2025
#

from sys import info, simdwidthof


@parameter
fn optimal_simd_width[dtype: DType]() -> Int:
    @parameter
    if info.is_apple_silicon():
        return 4 * simdwidthof[dtype]()
    else:
        return 2 * simdwidthof[dtype]()
