#
# system_utilities.mojo
# mosaic
#
# Created by Christian Bator on 02/02/2025
#

from sys import info, simdwidthof, exit


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
        fatal_error("Unsupported os for dynamic library filepath determination")
        while True:
            pass


@no_inline
fn fatal_error(error: Error):
    print(error)
    exit(1)


@no_inline
fn fatal_error[*Ts: Writable](*messages: *Ts):
    print(String(messages), flush=True)
    exit(1)
