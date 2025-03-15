#
# unsafe_number_pointer.mojo
# mosaic
#
# Created by Christian Bator on 03/06/2025
#

from memory import UnsafePointer, memset_zero


#
# UnsafeNumberPointer
#
@value
struct UnsafeNumberPointer[dtype: DType, *, complex: Bool]:
    var _data: UnsafePointer[Scalar[dtype]]

    #
    # Initialization
    #
    fn __init__(out self, count: Int):
        @parameter
        if complex:
            self._data = UnsafePointer[Scalar[dtype]].alloc(count * 2)
            memset_zero(self._data, count * 2)
        else:
            self._data = UnsafePointer[Scalar[dtype]].alloc(count)
            memset_zero(self._data, count)

    @implicit
    fn __init__(out self, owned data: UnsafePointer[Scalar[dtype]]):
        self._data = data

    fn __del__(owned self):
        self._data.free()

    fn unsafe_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self._data

    #
    # Access
    #
    fn __getitem__(self, index: Int) -> ScalarNumber[dtype, complex=complex]:
        return self.load[1](index)

    fn __setitem__(
        mut self: UnsafeNumberPointer[dtype, complex=complex],
        index: Int,
        value: ScalarNumber[dtype, complex=complex],
    ):
        self.store(index=index, value=value)

    fn load[
        width: Int
    ](self, index: Int) -> Number[dtype, width, complex=complex]:
        @parameter
        if complex:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](
                    self._data.offset(index * 2).load[width = 2 * width]()
                )
            )
        else:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](
                    self._data.offset(index).load[width=width]()
                )
            )

    fn store[
        width: Int, //
    ](mut self, index: Int, value: Number[dtype, width, complex=complex]):
        @parameter
        if complex:
            self._data.offset(index * 2).store(value.value)
        else:
            self._data.offset(index).store(value.value)

    fn strided_load[
        width: Int
    ](self, index: Int, stride: Int) -> Number[dtype, width, complex=complex]:
        @parameter
        if complex:
            return Number[dtype, width, complex=complex](
                real=self._data.offset(index * 2).strided_load[width=width](
                    stride * 2
                ),
                imaginary=self._data.offset(index * 2 + 1).strided_load[
                    width=width
                ](stride * 2),
            )
        else:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](
                    self._data.offset(index).strided_load[width=width](stride)
                )
            )

    fn strided_store[
        width: Int, //
    ](
        mut self,
        index: Int,
        stride: Int,
        value: Number[dtype, width, complex=complex],
    ):
        @parameter
        if complex:
            self._data.offset(index * 2).strided_store(
                val=value.real().value, stride=stride * 2
            )
            self._data.offset(index * 2 + 1).strided_store(
                val=value.imaginary().value, stride=stride * 2
            )
        else:
            self._data.offset(index).strided_store(
                val=value.value, stride=stride
            )

    fn gather[
        width: Int, //
    ](
        self,
        index: Int,
        offset_vector: SIMD[DType.index, width],
        mask_vector: SIMD[DType.bool, width],
    ) -> Number[dtype, width, complex=complex]:
        @parameter
        if complex:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](
                    self._data.offset(index * 2).gather(
                        offset=(offset_vector * 2).interleave(
                            offset_vector * 2 + 1
                        ),
                        mask=mask_vector.interleave(mask_vector),
                    )
                )
            )
        else:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](
                    self._data.offset(index).gather(
                        offset=offset_vector, mask=mask_vector
                    )
                )
            )

    fn scatter[
        width: Int, //
    ](
        self,
        index: Int,
        value: Number[dtype, width, complex=complex],
        offset_vector: SIMD[DType.index, width],
        mask_vector: SIMD[DType.bool, width],
    ):
        @parameter
        if complex:
            self._data.offset(index * 2).scatter(
                offset=(offset_vector * 2).interleave(offset_vector * 2 + 1),
                val=rebind[SIMD[dtype, width * 2]](value.value),
                mask=mask_vector.interleave(mask_vector),
            )
        else:
            self._data.offset(index).scatter(
                offset=offset_vector,
                val=rebind[SIMD[dtype, width]](value.value),
                mask=mask_vector,
            )
