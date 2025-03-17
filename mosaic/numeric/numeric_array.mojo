#
# numeric_array.mojo
# mosaic
#
# Created by Christian Bator on 03/06/2025
#

from memory import UnsafePointer, memset_zero, memcpy
from sys.intrinsics import likely, unlikely
from random import rand


#
# NumericArray
#
struct NumericArray[dtype: DType, *, complex: Bool = False](ExplicitlyCopyable, Sized):
    #
    # Fields
    #
    var _data: UnsafePointer[Scalar[dtype]]
    var _count: Int

    @always_inline
    fn _scalar_count(self) -> Int:
        @parameter
        if complex:
            return self._count * 2
        else:
            return self._count

    #
    # Initialization
    #
    fn __init__(out self, *, count: Int):
        debug_assert[assert_mode="safe"](count > 0)

        @parameter
        if complex:
            self._data = UnsafePointer[Scalar[dtype]].alloc(count * 2)
            memset_zero(self._data, count * 2)
        else:
            self._data = UnsafePointer[Scalar[dtype]].alloc(count)
            memset_zero(self._data, count)

        self._count = count

    fn __init__(out self, *values: ScalarNumber[dtype, complex=complex]):
        self = Self(values)

    fn __init__(out self, values: VariadicList[ScalarNumber[dtype, complex=complex]]):
        self = Self(count=len(values))

        for index in range(len(values)):
            self._unsafe_store(values[index], index=index)

    fn __init__(out self, owned values: List[ScalarNumber[dtype, complex=complex]]):
        self._data = values.steal_data().bitcast[Scalar[dtype]]()
        self._count = len(values)

    fn __init__(out self, owned data: UnsafePointer[ScalarNumber[dtype, complex=complex]], count: Int):
        debug_assert[assert_mode="safe"](count > 0)

        self._data = data.bitcast[Scalar[dtype]]()
        self._count = count

    fn __init__(out self, owned data: UnsafePointer[Scalar[dtype]], count: Int):
        debug_assert[assert_mode="safe"](count > 0)

        self._data = data
        self._count = count

    fn __moveinit__(out self, owned existing: Self):
        self._data = existing._data
        self._count = existing._count

    fn __del__(owned self):
        self._data.free()

    @staticmethod
    fn ascending(*, count: Int) -> Self:
        debug_assert[assert_mode="safe"](count > 0)

        var result = Self(count=count)

        for i in range(count):

            @parameter
            if complex:
                result._unsafe_store(ScalarNumber[dtype, complex=complex](real=i, imaginary=0), index=i)
            else:
                result._unsafe_store(ScalarNumber[dtype, complex=complex](i), index=i)

        return result^

    @staticmethod
    fn random(*, count: Int, min: Scalar[dtype] = Scalar[dtype].MIN_FINITE, max: Scalar[dtype] = Scalar[dtype].MAX_FINITE) -> Self:
        debug_assert[assert_mode="safe"](count > 0)

        var result = Self(count=count)
        rand(result.unsafe_data_ptr(), result._scalar_count(), min=min.cast[DType.float64](), max=max.cast[DType.float64]())

        return result^

    #
    # Access
    #
    @always_inline
    fn __getitem__(self, index: Int) raises -> ScalarNumber[dtype, complex=complex]:
        return self.load[1](index)

    @always_inline
    fn __setitem__(mut self: NumericArray[dtype, complex=complex], index: Int, value: ScalarNumber[dtype, complex=complex]) raises:
        self.store(value, index=index)

    @always_inline
    fn load[width: Int](self, index: Int) raises -> Number[dtype, width, complex=complex]:
        var verified_index = self._verified_index[width](index, stride=1)

        @parameter
        if complex:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](self._data.offset(verified_index * 2).load[width = 2 * width]())
            )
        else:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](self._data.offset(verified_index).load[width=width]())
            )

    @always_inline
    fn store[width: Int, //](mut self, value: Number[dtype, width, complex=complex], index: Int) raises:
        var verified_index = self._verified_index[width](index, stride=1)

        @parameter
        if complex:
            self._data.offset(verified_index * 2).store(value.value)
        else:
            self._data.offset(verified_index).store(value.value)

    @always_inline
    fn strided_load[width: Int](self, index: Int, stride: Int) raises -> Number[dtype, width, complex=complex]:
        var verified_index = self._verified_index[width](index, stride=stride)

        @parameter
        if complex:
            return Number[dtype, width, complex=complex](
                real=self._data.offset(verified_index * 2).strided_load[width=width](stride * 2),
                imaginary=self._data.offset(verified_index * 2 + 1).strided_load[width=width](stride * 2),
            )
        else:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](self._data.offset(verified_index).strided_load[width=width](stride))
            )

    @always_inline
    fn strided_store[width: Int, //](mut self, value: Number[dtype, width, complex=complex], index: Int, stride: Int) raises:
        var verified_index = self._verified_index[width](index, stride=stride)

        @parameter
        if complex:
            self._data.offset(verified_index * 2).strided_store(value.real().value, stride=stride * 2)
            self._data.offset(verified_index * 2 + 1).strided_store(value.imaginary().value, stride=stride * 2)
        else:
            self._data.offset(verified_index).strided_store(value.value, stride=stride)

    @always_inline
    fn gather[
        width: Int, //
    ](self, index: Int, offset_vector: SIMD[DType.index, width], mask_vector: SIMD[DType.bool, width]) raises -> Number[dtype, width, complex=complex]:
        var verified_index = self._verified_index_with_offset_vector[width](index, offset_vector=offset_vector)

        @parameter
        if complex:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](
                    self._data.offset(verified_index * 2).gather(
                        offset=(offset_vector * 2).interleave(offset_vector * 2 + 1),
                        mask=mask_vector.interleave(mask_vector),
                    )
                )
            )
        else:
            return Number[dtype, width, complex=complex](
                rebind[Number[dtype, width, complex=complex].Value](self._data.offset(verified_index).gather(offset=offset_vector, mask=mask_vector))
            )

    @always_inline
    fn scatter[
        width: Int, //
    ](self, value: Number[dtype, width, complex=complex], index: Int, offset_vector: SIMD[DType.index, width], mask_vector: SIMD[DType.bool, width]) raises:
        var verified_index = self._verified_index_with_offset_vector[width](index, offset_vector=offset_vector)

        @parameter
        if complex:
            self._data.offset(verified_index * 2).scatter(
                offset=(offset_vector * 2).interleave(offset_vector * 2 + 1),
                val=rebind[SIMD[dtype, width * 2]](value.value),
                mask=mask_vector.interleave(mask_vector),
            )
        else:
            self._data.offset(verified_index).scatter(offset=offset_vector, val=rebind[SIMD[dtype, width]](value.value), mask=mask_vector)

    #
    # Private Access
    #
    @always_inline
    fn _unsafe_store[width: Int, //](mut self, value: Number[dtype, width, complex=complex], index: Int):
        @parameter
        if complex:
            self._data.offset(index * 2).store(value.value)
        else:
            self._data.offset(index).store(value.value)

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
    # Bounds Checking
    #
    @always_inline
    fn _verified_index[width: Int](self, index: Int, stride: Int) raises -> Int:
        var shifted_index: Int

        if unlikely(index < 0):
            shifted_index = index + self._count
        else:
            shifted_index = index

        @parameter
        if width == 1:
            if likely(0 <= shifted_index < self._count):
                return shifted_index
            else:
                raise Error(
                    "Out of bounds NumericArray access: index = ",
                    index,
                    ", width = ",
                    width,
                    ", stride = ",
                    stride,
                    ", valid index range [",
                    -self._count,
                    ", ",
                    self._count,
                    ")",
                )
        else:
            if likely((0 <= shifted_index < self._count) and ((shifted_index + (stride - 1) * width) < self._count)):
                return shifted_index
            else:
                raise Error(
                    "Out of bounds NumericArray access: index = ",
                    index,
                    ", width = ",
                    width,
                    ", stride = ",
                    stride,
                    ", valid index range [",
                    -self._count,
                    ", ",
                    self._count,
                    ")",
                )

    @always_inline
    fn _verified_index_with_offset_vector[width: Int](self, index: Int, offset_vector: SIMD[DType.index, width]) raises -> Int:
        var shifted_index: Int

        if unlikely(index < 0):
            shifted_index = index + self._count
        else:
            shifted_index = index

        var min_offset = offset_vector.reduce_min()
        var max_offset = offset_vector.reduce_max()

        if likely(((shifted_index + min_offset) >= 0) and ((shifted_index + max_offset) < self._count)):
            return shifted_index
        else:
            raise Error(
                "Out of bounds NumericArray access: index = ",
                index,
                ", min_offset_index = ",
                shifted_index + min_offset,
                ", max_offset_index = ",
                index + max_offset,
                ", valid index range [",
                -self._count,
                ", ",
                self._count,
                ")",
            )

    #
    # ExplicitlyCopyable
    #
    fn copy(self) -> Self:
        var result = Self(count=self._count)
        memcpy(dest=result.unsafe_data_ptr(), src=self.unsafe_data_ptr(), count=self._scalar_count())

        return result^

    fn copy_into(self, mut other: Self):
        debug_assert[assert_mode="safe"](
            len(self) == len(other), "Invalid destination size provided to NumericArray copy_into(), ", len(other), " != ", len(self)
        )

        memcpy(dest=other.unsafe_data_ptr(), src=self.unsafe_data_ptr(), count=self._scalar_count())

    #
    # Unsafe Type Conversion
    #
    fn unsafe_bitcast[new_dtype: DType](owned self) -> NumericArray[new_dtype, complex=complex]:
        return NumericArray[new_dtype, complex=complex](self._data.bitcast[Scalar[new_dtype]](), count=self._count)

    #
    # Trait Implementations
    #
    @always_inline
    fn __len__(self) -> Int:
        return self._count
