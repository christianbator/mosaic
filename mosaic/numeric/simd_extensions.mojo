#
# simd_extensions.mojo
# mosaic
#
# Created by Christian Bator on 03/20/2025
#

"""SIMD helpers."""


def SIMDRange[width: Int, *, start: Int = 0, stride: Int = 1]() -> SIMD[DType.int, width]:
    """Returns the vector `[start, start + stride, start + 2 * stride, ...]` with `width` elements."""
    var result = SIMD[DType.int, width]()
    for i in range(width):
        result[i] = start + i * stride

    return result
