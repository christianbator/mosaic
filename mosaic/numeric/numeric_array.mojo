#
# numeric_array.mojo
# mosaic
#
# Created by Christian Bator on 03/06/2025
#

from std.memory import unsafe_memset_zero, unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.random import rand

from mosaic.utility import _assert

from .number import Number, ScalarNumber


#
# NumericArray
#
struct NumericArray[dtype: DType, *, complex: Bool = False](Copyable, Sized, Writable):
    #
    # Fields
    #
    var _data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin]
    var _count: Int

    #
    # Initialization
    #
    def __init__(out self, *, count: Int):
        _assert(count > 0, "Count must be greater than 0")

        comptime if Self.complex:
            self._data = unsafe_alloc[Scalar[Self.dtype]](2 * count)
            unsafe_memset_zero(self._data, 2 * count)
        else:
            self._data = unsafe_alloc[Scalar[Self.dtype]](count)
            unsafe_memset_zero(self._data, count)

        self._count = count

    def __init__(out self, *values: ScalarNumber[Self.dtype, complex=Self.complex]):
        self = Self(values)

    def __init__(out self, values: VariadicList[ScalarNumber[Self.dtype, complex=Self.complex], _]):
        self = Self(count=len(values))

        for index in range(len(values)):
            self.store(values[index], index=index)

    def __init__(out self, var values: List[ScalarNumber[Self.dtype, complex=Self.complex]]):
        self._count = len(values)
        self._data = values.unsafe_take_allocation().unsafe_leak().unsafe_bitcast[Scalar[Self.dtype]]()

    def __init__(out self, var data: Pointer[ScalarNumber[Self.dtype, complex=Self.complex], MutUntrackedOrigin], count: Int):
        _assert(count > 0, "Count must be greater than 0")

        self._data = data.unsafe_bitcast[Scalar[Self.dtype]]()
        self._count = count

    def __init__(out self, var data: Pointer[Scalar[Self.dtype], MutUntrackedOrigin], count: Int):
        _assert(count > 0, "Count must be greater than 0")

        self._data = data
        self._count = count

    def __init__(out self, *, deinit move: Self):
        self._data = move._data
        self._count = move._count

    def __deinit__(deinit self):
        self._data.unsafe_free()

    @staticmethod
    def ascending(*, count: Int) -> Self:
        _assert(count > 0, "Count must be greater than 0")

        var result = Self(count=count)

        for i in range(count):
            comptime if Self.complex:
                result.store(ScalarNumber[Self.dtype, complex=Self.complex](real=Scalar[Self.dtype](i), imaginary=0), index=i)
            else:
                result.store(ScalarNumber[Self.dtype, complex=Self.complex](i), index=i)

        return result^

    @staticmethod
    def random(
        *, count: Int, min: Scalar[Self.dtype] = Scalar[Self.dtype].MIN_FINITE, max: Scalar[Self.dtype] = Scalar[Self.dtype].MAX_FINITE
    ) -> Self:
        _assert(count > 0, "Count must be greater than 0")

        var result = Self(count=count)
        rand(result.unsafe_data_ptr(), result._scalar_count(), min=min.cast[DType.float64](), max=max.cast[DType.float64]())

        return result^

    #
    # Properties
    #
    @always_inline
    def _scalar_count(self) -> Int:
        comptime if Self.complex:
            return 2 * self._count
        else:
            return self._count

    #
    # Access
    #
    @always_inline
    def __getitem__(self, index: Int) -> ScalarNumber[Self.dtype, complex=Self.complex]:
        return self.load(index)

    @always_inline
    def __setitem__(mut self, index: Int, value: ScalarNumber[Self.dtype, complex=Self.complex]):
        self.store(value, index=index)

    @always_inline
    def load[width: Int = 1](self, index: Int) -> Number[Self.dtype, width, complex=Self.complex]:
        comptime if Self.complex:
            return Number[Self.dtype, width, complex=Self.complex](
                rebind[Number[Self.dtype, width, complex=Self.complex].Value](
                    self._data.unsafe_offset(index * 2).unsafe_load[width=2 * width]()
                )
            )
        else:
            return Number[Self.dtype, width, complex=Self.complex](
                rebind[Number[Self.dtype, width, complex=Self.complex].Value](self._data.unsafe_offset(index).unsafe_load[width=width]())
            )

    @always_inline
    def store[width: Int, //](mut self, value: Number[Self.dtype, width, complex=Self.complex], index: Int):
        comptime if Self.complex:
            self._data.unsafe_offset(index * 2).unsafe_store(value.value)
        else:
            self._data.unsafe_offset(index).unsafe_store(value.value)

    @always_inline
    def strided_load[width: Int = 1](self, index: Int, stride: Int) -> Number[Self.dtype, width, complex=Self.complex]:
        comptime if Self.complex:
            return Number[Self.dtype, width, complex=Self.complex](
                real=self._data.unsafe_offset(index * 2).unsafe_strided_load[width=width](stride * 2),
                imaginary=self._data.unsafe_offset(index * 2 + 1).unsafe_strided_load[width=width](stride * 2),
            )
        else:
            return Number[Self.dtype, width, complex=Self.complex](
                rebind[Number[Self.dtype, width, complex=Self.complex].Value](
                    self._data.unsafe_offset(index).unsafe_strided_load[width=width](stride)
                )
            )

    @always_inline
    def strided_store[width: Int, //](mut self, value: Number[Self.dtype, width, complex=Self.complex], index: Int, stride: Int):
        comptime if Self.complex:
            self._data.unsafe_offset(index * 2).unsafe_strided_store(value.real(), stride=stride * 2)
            self._data.unsafe_offset(index * 2 + 1).unsafe_strided_store(value.imaginary(), stride=stride * 2)
        else:
            self._data.unsafe_offset(index).unsafe_strided_store(value.value, stride=stride)

    @always_inline
    def gather[
        width: Int, //
    ](self, index: Int, offset: SIMD[DType.int, width], mask: SIMD[DType.bool, width]) -> Number[Self.dtype, width, complex=Self.complex]:
        comptime if Self.complex:
            return Number[Self.dtype, width, complex=Self.complex](
                rebind[Number[Self.dtype, width, complex=Self.complex].Value](
                    self._data.unsafe_offset(index * 2).unsafe_gather(
                        offset=(offset * 2).interleave(offset * 2 + 1),
                        mask=mask.interleave(mask),
                    )
                )
            )
        else:
            return Number[Self.dtype, width, complex=Self.complex](
                rebind[Number[Self.dtype, width, complex=Self.complex].Value](
                    self._data.unsafe_offset(index).unsafe_gather(offset=offset, mask=mask)
                )
            )

    @always_inline
    def scatter[
        width: Int, //
    ](
        self,
        value: Number[Self.dtype, width, complex=Self.complex],
        index: Int,
        offset: SIMD[DType.int, width],
        mask: SIMD[DType.bool, width],
    ):
        comptime if Self.complex:
            self._data.unsafe_offset(index * 2).unsafe_scatter(
                offset=(offset * 2).interleave(offset * 2 + 1),
                val=rebind[SIMD[Self.dtype, 2 * width]](value.value),
                mask=mask.interleave(mask),
            )
        else:
            self._data.unsafe_offset(index).unsafe_scatter(offset=offset, val=rebind[SIMD[Self.dtype, width]](value.value), mask=mask)

    #
    # Unsafe Access
    #
    @always_inline
    def unsafe_data_ptr(self) -> Pointer[Scalar[Self.dtype], MutUntrackedOrigin]:
        return self._data

    @always_inline
    def unsafe_uint8_ptr(self) -> Pointer[UInt8, MutUntrackedOrigin]:
        return self._data.unsafe_bitcast[UInt8]()

    #
    # Copyable
    #
    def __init__(out self, *, copy: Self):
        self = Self(count=copy._count)
        unsafe_memcpy(dest=self.unsafe_data_ptr(), src=copy.unsafe_data_ptr(), count=copy._scalar_count())

    def copy_into(self, mut other: Self):
        _assert(len(self) == len(other), "Invalid destination size provided to NumericArray copy_into(), ", len(other), " != ", len(self))

        unsafe_memcpy(dest=other.unsafe_data_ptr(), src=self.unsafe_data_ptr(), count=self._scalar_count())

    #
    # Sized
    #
    @always_inline
    def __len__(self) -> Int:
        return self._count

    #
    # Writable
    #
    def write_to[W: Writer](self, mut writer: W):
        writer.write("[NumericArray: [")

        for i in range(self._count):
            writer.write(self[i])
            writer.write(", " if i < self._count - 1 else "")

        writer.write("]]")
