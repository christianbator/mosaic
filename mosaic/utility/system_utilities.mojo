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
    if info.os_is_linux():
        return name + ".so"
    elif info.os_is_macos():
        return name + ".dylib"
    elif info.os_is_windows():
        return name + ".dll"
    else:
        return abort[String]("Unsupported os for dynamic library filepath determination")
