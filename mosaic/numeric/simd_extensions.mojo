#
# simd_extensions.mojo
# mosaic
#
# Created by Christian Bator on 03/20/2025
#


def SIMDRange[width: Int, *, start: Int = 0, stride: Int = 1]() -> SIMD[DType.int, width]:
    var result = SIMD[DType.int, width]()
    for i in range(width):
        result[i] = start + i * stride

    return result
