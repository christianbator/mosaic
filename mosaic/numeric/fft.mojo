#
# fft.mojo
# mosaic
#
# Created by Christian Bator on 04/01/2025
#

from memory import UnsafePointer
from collections.string import StaticString
from sys.ffi import _Global, _OwnedDLHandle, _get_dylib_function, c_int

from mosaic.utility import dynamic_library_filepath, fatal_error


#
# Backend
#
alias _libfft = _Global["libfft", _OwnedDLHandle, _load_libfft]()


fn _load_libfft() -> _OwnedDLHandle:
    return _OwnedDLHandle(dynamic_library_filepath("libmosaic-fft"))


alias _fft_func_name = "fft_" + String(fft_dtype)

#
# FFT
#
alias fft_dtype = DType.float32


fn fft[
    dtype: DType, depth: Int, complex: Bool, //, *, inverse: Bool = False
](matrix: Matrix[dtype, depth, complex=complex]) -> Matrix[fft_dtype, depth, complex=True]:
    var fft = _get_dylib_function[
        _libfft,
        _fft_func_name,
        fn (rows: c_int, cols: c_int, components: c_int, data_in: UnsafePointer[Scalar[fft_dtype]], data_out: UnsafePointer[Scalar[fft_dtype]], inverse: Bool),
    ]()

    var result = Matrix[fft_dtype, depth, complex=True](rows=matrix.rows(), cols=matrix.cols())

    @parameter
    if not complex and dtype != fft_dtype:
        var data_in = matrix.as_complex[fft_dtype]()

        fft(
            rows=matrix.rows(),
            cols=matrix.cols(),
            components=matrix.components(),
            data_in=data_in.unsafe_data_ptr(),
            data_out=result.unsafe_data_ptr(),
            inverse=inverse,
        )

        data_in.keep()

    elif not complex:
        var data_in = matrix.as_complex()

        fft(
            rows=matrix.rows(),
            cols=matrix.cols(),
            components=matrix.components(),
            data_in=data_in.unsafe_data_ptr().bitcast[Scalar[fft_dtype]](),
            data_out=result.unsafe_data_ptr(),
            inverse=inverse,
        )

        data_in.keep()

    elif dtype != fft_dtype:
        var data_in = matrix.astype[fft_dtype]()

        fft(
            rows=matrix.rows(),
            cols=matrix.cols(),
            components=matrix.components(),
            data_in=data_in.unsafe_data_ptr(),
            data_out=result.unsafe_data_ptr(),
            inverse=inverse,
        )

        data_in.keep()

    else:
        fft(
            rows=matrix.rows(),
            cols=matrix.cols(),
            components=matrix.components(),
            data_in=matrix.unsafe_data_ptr().bitcast[Scalar[fft_dtype]](),
            data_out=result.unsafe_data_ptr(),
            inverse=inverse,
        )

    return result^
