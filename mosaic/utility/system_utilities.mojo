#
# system_utilities.mojo
# mosaic
#
# Created by Christian Bator on 02/02/2025
#

"""Platform-specific constants and helpers."""

from std.os import abort
from std.sys import simd_width_of
from std.sys.info import CompilationTarget


def optimal_simd_width[dtype: DType]() -> Int:
    """Returns the SIMD width that performs best for `dtype` on the current platform."""
    comptime if CompilationTarget.is_macos():
        return 4 * simd_width_of[dtype]()
    else:
        return 2 * simd_width_of[dtype]()


comptime unroll_factor = 4
"""The unroll factor for vectorized loops."""


def dynamic_library_filepath(name: String) -> String:
    """Returns the platform's file name for the dynamic library `name`, such as `name.dylib` on macOS."""
    if CompilationTarget.is_linux():
        return name + ".so"
    elif CompilationTarget.is_macos():
        return name + ".dylib"
    else:
        abort("Unsupported os for dynamic library filepath determination")
