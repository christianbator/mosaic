#
# fft.mojo
# mosaic
#
# Created by Christian Bator on 04/01/2025
#

from std.os import abort
from std.ffi import _Global, OwnedDLHandle, _get_dylib_function, c_int

from mosaic.utility import dynamic_library_filepath

from .matrix import Matrix


#
# Backend
#
comptime _libfft = _Global["libfft", _load_libfft]()


def _load_libfft() -> OwnedDLHandle:
    try:
        return OwnedDLHandle(dynamic_library_filepath("libmosaic-fft"))
    except:
        abort()


comptime _fft_func_name = "fft_" + String(fft_dtype)

#
# FFT
#
comptime fft_dtype = DType.float32


def fft[
    dtype: DType, depth: Int, complex: Bool, //, *, inverse: Bool = False
](matrix: Matrix[dtype, depth, complex=complex]) -> Matrix[fft_dtype, depth, complex=True]:
    try:
        var fft = _get_dylib_function[
            _libfft,
            _fft_func_name,
            def(
                rows: c_int,
                cols: c_int,
                components: c_int,
                data_in: Pointer[Scalar[fft_dtype], MutUnsafeAnyOrigin],
                data_out: Pointer[Scalar[fft_dtype], MutUnsafeAnyOrigin],
                inverse: Bool,
            ) thin abi("C"),
        ]()

        var result = Matrix[fft_dtype, depth, complex=True](rows=matrix.rows(), cols=matrix.cols())

        comptime if not complex and dtype != fft_dtype:
            var data_in = matrix.as_complex[fft_dtype]()

            fft(
                rows=c_int(matrix.rows()),
                cols=c_int(matrix.cols()),
                components=c_int(matrix.components()),
                data_in=data_in.unsafe_data_ptr().as_unsafe_any_origin(),
                data_out=result.unsafe_data_ptr().as_unsafe_any_origin(),
                inverse=inverse,
            )

            data_in.keep()

        elif not complex:
            var data_in = matrix.as_complex()

            fft(
                rows=c_int(matrix.rows()),
                cols=c_int(matrix.cols()),
                components=c_int(matrix.components()),
                data_in=data_in.unsafe_data_ptr().unsafe_bitcast[Scalar[fft_dtype]]().as_unsafe_any_origin(),
                data_out=result.unsafe_data_ptr().as_unsafe_any_origin(),
                inverse=inverse,
            )

            data_in.keep()

        elif dtype != fft_dtype:
            var data_in = matrix.as_type[fft_dtype]()

            fft(
                rows=c_int(matrix.rows()),
                cols=c_int(matrix.cols()),
                components=c_int(matrix.components()),
                data_in=data_in.unsafe_data_ptr().as_unsafe_any_origin(),
                data_out=result.unsafe_data_ptr().as_unsafe_any_origin(),
                inverse=inverse,
            )

            data_in.keep()

        else:
            fft(
                rows=c_int(matrix.rows()),
                cols=c_int(matrix.cols()),
                components=c_int(matrix.components()),
                data_in=matrix.unsafe_data_ptr().unsafe_bitcast[Scalar[fft_dtype]]().as_unsafe_any_origin(),
                data_out=result.unsafe_data_ptr().as_unsafe_any_origin(),
                inverse=inverse,
            )

        return result^
    except:
        abort()
