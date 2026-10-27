#
# system_utilities.mojo
# mosaic
#
# Created by Christian Bator on 02/02/2025
#

from std.os import abort
from std.sys import simd_width_of
from std.sys.info import CompilationTarget


def optimal_simd_width[dtype: DType]() -> Int:
    comptime if CompilationTarget.is_macos():
        return 4 * simd_width_of[dtype]()
    else:
        return 2 * simd_width_of[dtype]()


comptime unroll_factor = 4


def dynamic_library_filepath(name: String) -> String:
    if CompilationTarget.is_linux():
        return name + ".so"
    elif CompilationTarget.is_macos():
        return name + ".dylib"
    else:
        abort("Unsupported os for dynamic library filepath determination")
