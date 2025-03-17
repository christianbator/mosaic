#
# system_utilities.mojo
# mosaic
#
# Created by Christian Bator on 02/02/2025
#

from os import abort
from sys import info, simdwidthof


@parameter
fn optimal_simd_width[dtype: DType]() -> Int:
    @parameter
    if info.is_apple_silicon():
        return 4 * simdwidthof[dtype]()
    else:
        return 2 * simdwidthof[dtype]()


alias unroll_factor = 4


@parameter
fn dynamic_library_filepath(name: String) -> String:
    alias prefix = "mosaic/"

    if info.os_is_linux():
        return prefix + name + ".so"
    elif info.os_is_macos():
        return prefix + name + ".dylib"
    elif info.os_is_windows():
        return prefix + name + ".dll"
    else:
        abort("Unsupported os for dynamic library filepath determination")
        while True:
            pass
