#
# numeric_array.mojo
# mosaic
#
# Created by Christian Bator on 03/06/2025
#

from memory import UnsafePointer, memset_zero, memcpy
from random import rand

from mosaic.utility import _assert


#
# NumericArray
#
struct NumericArray[dtype: DType, *, complex: Bool = False](ExplicitlyCopyable, Sized):
    #
    # Fields
    #
    var _data: UnsafePointer[Scalar[dtype]]
    var _count: Int

    #
    # Initialization
    #
    fn __init__(out self, *, count: Int):
        _assert(count > 0, "Count must be greater than 0")

        @parameter
        if complex:
            self._data = UnsafePointer[Scalar[dtype]].alloc(2 * count)
            memset_zero(self._data, 2 * count)
        else:
            self._data = UnsafePointer[Scalar[dtype]].alloc(count)
            memset_zero(self._data, count)

        self._count = count

    fn __init__(out self, *values: ScalarNumber[dtype, complex=complex]):
        self = Self(values)

    fn __init__(out self, values: VariadicList[ScalarNumber[dtype, complex=complex]]):
        self = Self(count=len(values))

        for index in range(len(values)):
            self.store(values[index], index=index)

    fn __init__(out self, owned values: List[ScalarNumber[dtype, complex=complex]]):
        self._data = values.steal_data().bitcast[Scalar[dtype]]()
        self._count = len(values)

    fn __init__(out self, owned data: UnsafePointer[ScalarNumber[dtype, complex=complex]], count: Int):
        _assert(count > 0, "Count must be greater than 0")

        self._data = data.bitcast[Scalar[dtype]]()
        self._count = count

    fn __init__(out self, owned data: UnsafePointer[Scalar[dtype]], count: Int):
        _assert(count > 0, "Count must be greater than 0")

        self._data = data
        self._count = count

    fn __moveinit__(out self, owned existing: Self):
        self._data = existing._data
        self._count = existing._count

    fn __del__(owned self):
        self._data.free()

    @staticmethod
    fn ascending(*, count: Int) -> Self:
        _assert(count > 0, "Count must be greater than 0")

        var result = Self(count=count)

        for i in range(count):

            @parameter
            if complex:
                result.store(ScalarNumber[dtype, complex=complex](real=i, imaginary=0), index=i)
            else:
                result.store(ScalarNumber[dtype, complex=complex](i), index=i)

        return result^

    @staticmethod
    fn random(*, count: Int, min: Scalar[dtype] = Scalar[dtype].MIN_FINITE, max: Scalar[dtype] = Scalar[dtype].MAX_FINITE) -> Self:
        _assert(count > 0, "Count must be greater than 0")

        var result = Self(count=count)
        rand(result.unsafe_data_ptr(), result._scalar_count(), min=min.cast[DType.float64](), max=max.cast[DType.float64]())

        return result^

    #
    # Properties
    #
    @always_inline
    fn _scalar_count(self) -> Int:
        @parameter
        if complex:
            return 2 * self._count
        else:
            return self._count

    #
    # Access
    #
    @always_inline
    fn __getitem__(self, index: Int) -> ScalarNumber[dtype, complex=complex]:
        return self.load(index)

    @always_inline
    fn __setitem__(mut self: NumericArray[dtype, complex=complex], index: Int, value: ScalarNumber[dtype, complex=complex]):
        self.store(value, index=index)

    @always_inline
    fn load[width: Int = 1](self, index: Int) -> Number[dtype, width, complex=complex]:
        @parameter
        if complex:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](self._data.offset(index * 2).load[width = 2 * width]())
            )
        else:
            return Number[dtype, width, complex=complex](rebind[Number[dtype, width, complex=complex].Value](self._data.offset(index).load[width=width]()))

    @always_inline
    fn store[width: Int, //](mut self, value: Number[dtype, width, complex=complex], index: Int):
        @parameter
        if complex:
            self._data.offset(index * 2).store(value.value)
        else:
            self._data.offset(index).store(value.value)

    @always_inline
    fn strided_load[width: Int = 1](self, index: Int, stride: Int) -> Number[dtype, width, complex=complex]:
        @parameter
        if complex:
            return Number[dtype, width, complex=complex](
                real=self._data.offset(index * 2).strided_load[width=width](stride * 2),
                imaginary=self._data.offset(index * 2 + 1).strided_load[width=width](stride * 2),
            )
        else:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](self._data.offset(index).strided_load[width=width](stride))
            )

    @always_inline
    fn strided_store[width: Int, //](mut self, value: Number[dtype, width, complex=complex], index: Int, stride: Int):
        @parameter
        if complex:
            self._data.offset(index * 2).strided_store(value.real().value, stride=stride * 2)
            self._data.offset(index * 2 + 1).strided_store(value.imaginary().value, stride=stride * 2)
        else:
            self._data.offset(index).strided_store(value.value, stride=stride)

    @always_inline
    fn gather[width: Int, //](self, index: Int, offset: SIMD[DType.index, width], mask: SIMD[DType.bool, width]) -> Number[dtype, width, complex=complex]:
        @parameter
        if complex:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](
                    self._data.offset(index * 2).gather(
                        offset=(offset * 2).interleave(offset * 2 + 1),
                        mask=mask.interleave(mask),
                    )
                )
            )
        else:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](self._data.offset(index).gather(offset=offset, mask=mask))
            )

    @always_inline
    fn scatter[width: Int, //](self, value: Number[dtype, width, complex=complex], index: Int, offset: SIMD[DType.index, width], mask: SIMD[DType.bool, width]):
        @parameter
        if complex:
            self._data.offset(index * 2).scatter(
                offset=(offset * 2).interleave(offset * 2 + 1),
                val=rebind[SIMD[dtype, 2 * width]](value.value),
                mask=mask.interleave(mask),
            )
        else:
            self._data.offset(index).scatter(offset=offset, val=rebind[SIMD[dtype, width]](value.value), mask=mask)

    #
    # Unsafe Access
    #
    @always_inline
    fn unsafe_data_ptr(self) -> UnsafePointer[Scalar[dtype]]:
        return self._data

    @always_inline
    fn unsafe_uint8_ptr(self) -> UnsafePointer[UInt8]:
        return self._data.bitcast[UInt8]()

    #
    # ExplicitlyCopyable
    #
    fn copy(self) -> Self:
        var result = Self(count=self._count)
        memcpy(dest=result.unsafe_data_ptr(), src=self.unsafe_data_ptr(), count=self._scalar_count())

        return result^

    fn copy_into(self, mut other: Self):
        _assert(len(self) == len(other), "Invalid destination size provided to NumericArray copy_into(), ", len(other), " != ", len(self))

        memcpy(dest=other.unsafe_data_ptr(), src=self.unsafe_data_ptr(), count=self._scalar_count())

    #
    # Sized
    #
    @always_inline
    fn __len__(self) -> Int:
        return self._count
