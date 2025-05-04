#
# number.mojo
# mosaic
#
# Created by Christian Bator on 03/04/2025
#

from math import sqrt, log, atan2, Ceilable, CeilDivable, Floorable, Truncable, isclose
from complex import ComplexSIMD
from builtin.dtype import _integral_type_of
from sys import is_big_endian
from collections import InlineArray
from utils import IndexList

#
# Type Aliases
#
alias ScalarNumber = Number[width=1, complex=_]

#
# Print Precision
#
alias print_precision = 3


#
# Number
#
@register_passable("trivial")
struct Number[dtype: DType, width: Int, *, complex: Bool = False](
    Absable,
    Boolable,
    Ceilable,
    CeilDivable,
    CollectionElement,
    ExplicitlyCopyable,
    Floatable,
    Floorable,
    Hashable,
    Intable,
    Indexer,
    Roundable,
    Stringable,
    Truncable,
    Writable,
):
    #
    # Fields
    #
    alias Value = SIMD[dtype, 2 * width if complex else width]

    alias MAX = Self._max()
    alias MIN = Self(Self.Value.MIN)
    alias MAX_FINITE = Self(Self.Value.MAX_FINITE)
    alias MIN_FINITE = Self(Self.Value.MIN_FINITE)

    var value: Self.Value

    #
    # Initialization
    #
    @always_inline
    fn __init__(out self):
        self.value = Self.Value(0)

    @always_inline
    @implicit
    fn __init__(out self, value: Self.Value):
        self.value = value

    @always_inline
    fn __init__[other_dtype: DType, //](out self, value: Number[other_dtype, width, complex=complex]):
        self.value = value.value.cast[dtype]()

    @always_inline
    @implicit
    fn __init__(out self, value: UInt):
        @parameter
        if complex:
            self.value = Self.Value(value, 0)
        else:
            self.value = value

    @always_inline
    @implicit
    fn __init__(out self, value: Int):
        @parameter
        if complex:
            self.value = Self.Value(value, 0)
        else:
            self.value = value

    @always_inline
    fn __init__[T: Floatable](out self: ScalarNumber[DType.float64, complex=complex], value: T):
        @parameter
        if complex:
            self.value = ScalarNumber[DType.float64, complex=complex].Value(value.__float__(), 0.0)
        else:
            self.value = value.__float__()

    @always_inline
    fn __init__[T: FloatableRaising](out self: ScalarNumber[DType.float64, complex=complex], value: T) raises:
        @parameter
        if complex:
            self.value = ScalarNumber[DType.float64, complex=complex].Value(value.__float__(), 0.0)
        else:
            self.value = value.__float__()

    @always_inline
    @implicit
    fn __init__(out self, value: IntLiteral):
        @parameter
        if complex:
            self.value = Self.Value(value, 0)
        else:
            self.value = value

    @always_inline
    @implicit
    fn __init__(out self, value: FloatLiteral):
        @parameter
        if complex:
            self.value = Self.Value(value, 0)
        else:
            self.value = value

    @always_inline
    @implicit
    fn __init__(out self: Number[DType.bool, width, complex=False], value: Bool, /):
        self.value = value

    @always_inline
    @implicit
    fn __init__(out self, value: ScalarNumber[dtype, complex=complex], /):
        @parameter
        if complex:
            self.value = rebind[Self.Value](SIMD[dtype, width](value.value[0]).interleave(SIMD[dtype, width](value.value[1])))
        else:
            self.value = value.value[0]

    @always_inline
    @implicit
    fn __init__(out self, *elems: Scalar[dtype]):
        # TODO: Make this a compile-time check when possible
        debug_assert(width == len(elems), "Mismatch in the number of elements in the Number variadic constructor")

        self.value = Self.Value(0)

        @parameter
        if complex:

            @parameter
            for i in range(0, 2 * width, 2):
                self.value[i] = elems[i]
        else:

            @parameter
            for i in range(width):
                self.value[i] = elems[i]

    @always_inline
    fn __init__(out self, real: SIMD[dtype, width], imaginary: SIMD[dtype, width]):
        constrained[complex, "__init__(real, imaginary) is only available for complex numbers"]()

        self.value = rebind[Self.Value](real.interleave(imaginary))

    @always_inline
    @implicit
    fn __init__(out self, tuple: (SIMD[dtype, width], SIMD[dtype, width])):
        constrained[complex, "__init__(tuple) is only available for complex numbers"]()

        self = Self(real=tuple[0], imaginary=tuple[1])

    @staticmethod
    fn from_bits[int_type: DType, //](value: SIMD[int_type, Self.Value.size]) -> Number[dtype, width, complex=complex]:
        return Self(Self.Value.from_bits(value))

    @staticmethod
    fn _max() -> Self:
        constrained[not complex, "MAX is only available for non-complex numbers"]()

        return Self(Self.Value.MAX)

    @staticmethod
    fn _min() -> Self:
        constrained[not complex, "MIN is only available for non-complex numbers"]()

        return Self(Self.Value.MIN)

    @staticmethod
    fn _max_finite() -> Self:
        constrained[not complex, "MAX_FINITE is only available for non-complex numbers"]()

        return Self(Self.Value.MAX_FINITE)

    @staticmethod
    fn _min_finite() -> Self:
        constrained[not complex, "MIN_FINITE is only available for non-complex numbers"]()

        return Self(Self.Value.MIN_FINITE)

    #
    # ExplicityCopyable
    #
    @always_inline
    fn copy(self) -> Self:
        return self

    #
    # Access
    #
    @always_inline
    fn __getitem__(self, index: Int) -> ScalarNumber[dtype, complex=complex]:
        @parameter
        if complex:
            return ScalarNumber[dtype, complex=complex](real=self.value[index], imaginary=self.value[index + 1])
        else:
            return ScalarNumber[dtype, complex=complex](self.value[index])

    @always_inline
    fn __setitem__(mut self, index: Int, value: ScalarNumber[dtype, complex=complex]):
        @parameter
        if complex:
            self.value[index] = value.value[0]
            self.value[index + 1] = value.value[1]
        else:
            self.value[index] = value.value[0]

    fn __contains__(self, value: ScalarNumber[dtype, complex=complex]) -> Bool:
        @parameter
        if complex:

            @parameter
            for i in range(width):
                if self[i] == value:
                    return True

            return False
        else:
            return self.value.__contains__(value.value[0])

    @always_inline
    fn real(self) -> SIMD[dtype, width]:
        constrained[complex, "real() is only available for complex numbers"]()

        return rebind[SIMD[dtype, width]](self.value.deinterleave()[0])

    @always_inline
    fn imaginary(self) -> SIMD[dtype, width]:
        constrained[complex, "imaginary() is only available for complex numbers"]()

        return rebind[SIMD[dtype, width]](self.value.deinterleave()[1])

    #
    # Operators
    #
    @always_inline
    fn __neg__(self) -> Self:
        return Self(-self.value)

    @always_inline
    fn __add__(self, rhs: Self) -> Self:
        return Self(self.value + rhs.value)

    @always_inline
    fn __radd__(self, lhs: Self) -> Self:
        return lhs + self

    @always_inline
    fn __iadd__(mut self, rhs: Self):
        self = self + rhs

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        return Self(self.value - rhs.value)

    @always_inline
    fn __rsub__(self, lhs: Self) -> Self:
        return lhs - self

    @always_inline
    fn __isub__(mut self, rhs: Self):
        self = self - rhs.value

    @always_inline
    fn __mul__(self, rhs: Self) -> Self:
        @parameter
        if complex:
            return Self(
                real=self.real() * rhs.real() - self.imaginary() * rhs.imaginary(),
                imaginary=self.real() * rhs.imaginary() + self.imaginary() * rhs.real(),
            )
        else:
            return Self(self.value * rhs.value)

    @always_inline
    fn __rmul__(self, lhs: Self) -> Self:
        return lhs * self

    @always_inline
    fn __imul__(mut self, rhs: Self):
        self = self * rhs

    @always_inline
    fn __truediv__(self, rhs: Self) -> Self:
        @parameter
        if complex:
            var denominator = rhs.squared_norm().value

            return Self(
                real=(self.real() * rhs.real() + self.imaginary() * rhs.imaginary()) / denominator,
                imaginary=(self.imaginary() * rhs.real() - self.real() * rhs.imaginary()) / denominator,
            )
        else:
            return Self(self.value / rhs.value)

    @always_inline
    fn __rtruediv__(self, lhs: Self) -> Self:
        return lhs / self

    @always_inline
    fn __itruediv__(mut self, rhs: Self):
        self = self / rhs

    @always_inline
    fn __floordiv__(self, rhs: Self) -> Self:
        constrained[not complex, "__floordiv__() is only available for non-complex numbers"]()

        return Self(self.value // rhs.value)

    @always_inline
    fn __rfloordiv__(self, lhs: Self) -> Self:
        return lhs // lhs

    @always_inline
    fn __ifloordiv__(mut self, rhs: Self):
        self = self // rhs

    @always_inline
    fn __mod__(self, rhs: Self) -> Self:
        constrained[not complex, "__mod__() is only available for non-complex numbers"]()

        return Self(self.value % rhs.value)

    @always_inline
    fn __rmod__(self, lhs: Self) -> Self:
        return lhs % self

    @always_inline
    fn __imod__(mut self, rhs: Self):
        self = self % rhs

    @always_inline
    fn __pow__(self, rhs: Self) -> Self:
        constrained[not complex, "__pow__() is only available for non-complex numbers"]()

        return Self(self.value**rhs.value)

    @always_inline
    fn __rpow__(self, lhs: Self) -> Self:
        return lhs**self

    @always_inline
    fn __ipow__(mut self, rhs: Self):
        self = self**rhs

    @always_inline
    fn __and__(self, rhs: Self) -> Self:
        constrained[not complex, "__and__() is only available for non-complex numbers"]()

        return Self(self.value & rhs.value)

    @always_inline
    fn __rand__(self, lhs: Self) -> Self:
        return lhs & self

    @always_inline
    fn __iand__(mut self, rhs: Self):
        self = self & rhs

    @always_inline
    fn __or__(self, rhs: Self) -> Self:
        constrained[not complex, "__or__() is only available for non-complex numbers"]()

        return Self(self.value | rhs.value)

    @always_inline
    fn __ror__(self, lhs: Self) -> Self:
        return lhs | self

    @always_inline
    fn __ior__(mut self, rhs: Self):
        self = self | rhs

    @always_inline
    fn __xor__(self, rhs: Self) -> Self:
        constrained[not complex, "__xor__() is only available for non-complex numbers"]()

        return Self(self.value ^ rhs.value)

    @always_inline
    fn __rxor__(self, lhs: Self) -> Self:
        return lhs ^ self

    @always_inline
    fn __ixor__(mut self, rhs: Self):
        self = self ^ rhs

    @always_inline
    fn __lshift__(self, rhs: Self) -> Self:
        constrained[not complex, "__lshift__() is only available for non-complex numbers"]()

        return Self(self.value << rhs.value)

    @always_inline
    fn __rlshift__(self, lhs: Self) -> Self:
        return lhs << self

    @always_inline
    fn __ilshift__(mut self, rhs: Self):
        self = self << rhs

    @always_inline
    fn __rshift__(self, rhs: Self) -> Self:
        constrained[not complex, "__rshift__() is only available for non-complex numbers"]()

        return Self(self.value >> rhs.value)

    @always_inline
    fn __rrshift__(self, lhs: Self) -> Self:
        return lhs >> self

    @always_inline
    fn __irshift__(mut self, rhs: Self):
        self = self >> rhs

    @always_inline
    fn __lt__(self, rhs: Self) -> SIMD[DType.bool, width]:
        constrained[not complex, "__lt__() is only available for non-complex numbers"]()

        return rebind[SIMD[DType.bool, width]](self.value < rhs.value)

    @always_inline
    fn __le__(self, rhs: Self) -> SIMD[DType.bool, width]:
        constrained[not complex, "__le__() is only available for non-complex numbers"]()

        return rebind[SIMD[DType.bool, width]](self.value <= rhs.value)

    @always_inline
    fn __eq__(self, rhs: Self) -> SIMD[DType.bool, width]:
        @parameter
        if complex:
            var result = SIMD[DType.bool, width]()

            @parameter
            for i in range(0, 2 * width, 2):
                result[i // 2] = self.value[i] == rhs.value[i] and self.value[i + 1] == rhs.value[i + 1]

            return result
        else:
            return rebind[SIMD[DType.bool, width]](self.value == rhs.value)

    @always_inline
    fn __ne__(self, rhs: Self) -> SIMD[DType.bool, width]:
        return not (self == rhs)

    @always_inline
    fn __gt__(self, rhs: Self) -> SIMD[DType.bool, width]:
        constrained[not complex, "__gt__() is only available for non-complex numbers"]()

        return rebind[SIMD[DType.bool, width]](self.value > rhs.value)

    @always_inline
    fn __ge__(self, rhs: Self) -> SIMD[DType.bool, width]:
        constrained[not complex, "__ge__() is only available for non-complex numbers"]()

        return rebind[SIMD[DType.bool, width]](self.value >= rhs.value)

    @always_inline
    fn __invert__(self) -> Self:
        constrained[not complex, "__invert__() is only available for non-complex numbers"]()

        return Self(self.value.__invert__())

    #
    # Numeric Traits
    #
    @always_inline
    fn __abs__(self) -> Self:
        constrained[not complex, "__abs__() is only available for non-complex numbers, see norm() instead"]()

        return Self(self.value.__abs__())

    @always_inline
    fn __bool__(self) -> Bool:
        constrained[not complex, "__bool__() is only available for non-complex numbers"]()

        return self.value.__bool__()

    @always_inline
    fn __ceil__(self) -> Self:
        constrained[not complex, "__ceil__() is only available for non-complex numbers"]()

        return Self(self.value.__ceil__())

    @always_inline
    fn __ceildiv__(self, denominator: Self) -> Self:
        constrained[not complex, "__ceildiv__() is only available for non-complex numbers"]()

        return Self(self.value.__ceildiv__(denominator.value))

    @always_inline
    fn __float__(self) -> Float64:
        constrained[not complex, "__float__() is only available for non-complex numbers"]()

        return self.value.__float__()

    @always_inline
    fn __floor__(self) -> Self:
        constrained[not complex, "__floor__() is only available for non-complex numbers"]()

        return Self(self.value.__floor__())

    fn __hash__(self) -> UInt:
        return self.value.__hash__()

    @always_inline
    fn __int__(self) -> Int:
        constrained[not complex, "__int__() is only available for non-complex numbers"]()

        return self.value.__int__()

    @always_inline
    fn __index__(self) -> __mlir_type.index:
        constrained[not complex, "__index__() is only available for non-complex numbers"]()

        return self.value.__index__()

    @always_inline
    fn __round__(self) -> Self:
        constrained[not complex, "__round__() is only available for non-complex numbers"]()

        return Self(self.value.__round__())

    @always_inline
    fn __round__(self, ndigits: Int) -> Self:
        constrained[not complex, "__round__(ndigits) is only available for non-complex numbers"]()

        return Self(self.value.__round__(ndigits))

    @always_inline
    fn __trunc__(self) -> Self:
        constrained[not complex, "__trunc__() is only available for non-complex numbers"]()

        return Self(self.value.__trunc__())

    #
    # Numeric Methods
    #
    @always_inline
    fn to_bits[int_dtype: DType = _integral_type_of[dtype]()](self) -> Number[int_dtype, width, complex=complex]:
        return Number[int_dtype, width, complex=complex](self.value.to_bits())

    @staticmethod
    fn from_bytes[big_endian: Bool = is_big_endian()](bytes: InlineArray[Byte, dtype.sizeof()]) -> Self:
        constrained[width == 1 and not complex, "from_bytes() is only available for scalar, non-complex numbers"]()

        return Self(Scalar[dtype].from_bytes(bytes))

    fn as_bytes[big_endian: Bool = is_big_endian()](self) -> InlineArray[Byte, dtype.sizeof()]:
        constrained[width == 1 and not complex, "as_bytes() is only available for scalar, non-complex numbers"]()

        return self.value.as_bytes()

    fn clamp(self, lower_bound: Number[dtype, width, complex=False], upper_bound: Number[dtype, width, complex=False]) -> Self:
        constrained[not complex, "clamp() is only available for non-complex numbers"]()

        return Self(self.value.clamp(lower_bound=rebind[Self.Value](lower_bound.value), upper_bound=rebind[Self.Value](upper_bound.value)))

    @always_inline
    fn fma(self, multiplier: Self, accumulator: Self) -> Self:
        @parameter
        if complex:
            var result = ComplexSIMD(re=self.real(), im=self.imaginary()).fma(
                b=ComplexSIMD(re=multiplier.real(), im=multiplier.imaginary()),
                c=ComplexSIMD(re=accumulator.real(), im=accumulator.imaginary()),
            )

            return Self(real=result.re, imaginary=result.im)
        else:
            return Self(self.value.fma(multiplier=multiplier.value, accumulator=accumulator.value))

    @always_inline
    fn shuffle[*mask: Int](self) -> Self:
        return self.shuffle[*mask](self)

    @always_inline
    fn shuffle[*mask: Int](self, other: Self) -> Self:
        @parameter
        if complex:
            return Self(real=self.real().shuffle[*mask](other.real()), imaginary=self.imaginary().shuffle[*mask](other.imaginary()))
        else:
            return Self(self.value.shuffle[*mask](other.value))

    @always_inline
    fn shuffle[mask: IndexList[width, **_]](self) -> Self:
        return self.shuffle[mask=mask](self)

    @always_inline
    fn shuffle[mask: IndexList[width, **_]](self, other: Self) -> Self:
        @parameter
        if complex:
            return Self(real=self.real().shuffle[mask=mask](other.real()), imaginary=self.imaginary().shuffle[mask=mask](other.imaginary()))
        else:
            return Self(rebind[Self.Value](rebind[SIMD[dtype, width]](self.value).shuffle[mask=mask](rebind[SIMD[dtype, width]](other.value))))

    @always_inline
    fn slice[output_width: Int, /, *, offset: Int = 0](self) -> Number[dtype, output_width, complex=complex]:
        return Number[dtype, output_width, complex=complex](
            rebind[Number[dtype, output_width, complex=complex].Value](
                self.value.slice[2 * output_width if complex else output_width, offset = 2 * offset if complex else offset]()
            )
        )

    @always_inline
    fn insert[*, offset: Int = 0](self, value: Number[dtype, _, complex=complex]) -> Self:
        return Self(self.value.insert[offset = 2 * offset if complex else offset](value.value))

    @always_inline
    fn join(self, other: Self, out result: Number[dtype, 2 * width, complex=complex]):
        result = Number[dtype, 2 * width, complex=complex](rebind[Number[dtype, 2 * width, complex=complex].Value](self.value.join(other.value)))

    @always_inline
    fn interleave(self, other: Self) -> Number[dtype, 2 * width, complex=complex]:
        @parameter
        if complex:
            return Number[dtype, 2 * width, complex=complex](
                real=self.real().interleave(other.real()), imaginary=self.imaginary().interleave(other.imaginary())
            )
        else:
            return Number[dtype, 2 * width, complex=complex](rebind[Number[dtype, 2 * width, complex=complex].Value](self.value.interleave(other.value)))

    @always_inline
    fn split(self) -> (Number[dtype, width // 2, complex=complex], Number[dtype, width // 2, complex=complex]):
        var split = self.value.split()

        return (
            Number[dtype, width // 2, complex=complex](rebind[Number[dtype, width // 2, complex=complex].Value](split[0])),
            Number[dtype, width // 2, complex=complex](rebind[Number[dtype, width // 2, complex=complex].Value](split[1])),
        )

    @always_inline
    fn deinterleave(self) -> (Number[dtype, width // 2, complex=complex], Number[dtype, width // 2, complex=complex]):
        @parameter
        if complex:
            var real = self.real().deinterleave()
            var imaginary = self.imaginary().deinterleave()

            return (
                Number[dtype, width // 2, complex=complex](real=real[0], imaginary=imaginary[0]),
                Number[dtype, width // 2, complex=complex](real=real[1], imaginary=imaginary[1]),
            )
        else:
            var deinterleaved = self.value.deinterleave()

            return (
                Number[dtype, width // 2, complex=complex](rebind[Number[dtype, width // 2, complex=complex].Value](deinterleaved[0])),
                Number[dtype, width // 2, complex=complex](rebind[Number[dtype, width // 2, complex=complex].Value](deinterleaved[1])),
            )

    @always_inline
    fn reduce[
        func: fn[dtype: DType, width: Int] (Number[dtype, width, complex=complex], Number[dtype, width, complex=complex]) capturing -> Number[
            dtype, width, complex=complex
        ],
        size_out: Int = 1,
    ](self) -> Number[dtype, size_out, complex=complex]:
        constrained[size_out <= width, "Reduce must specify width less than original number width"]()

        @parameter
        if width == size_out:
            return rebind[Number[dtype, size_out, complex=complex]](self)
        else:
            var lhs: Number[dtype, width // 2, complex=complex]
            var rhs: Number[dtype, width // 2, complex=complex]
            lhs, rhs = self.split()

            return func(lhs, rhs).reduce[func, size_out]()

    @always_inline
    fn reduce_max[size_out: Int = 1](self) -> Number[dtype, size_out, complex=complex]:
        constrained[not complex, "reduce_max() is only available for non-complex numbers"]()

        return Number[dtype, size_out, complex=complex](rebind[Number[dtype, size_out, complex=complex].Value](self.value.reduce_max[size_out]()))

    @always_inline
    fn reduce_min[size_out: Int = 1](self) -> Number[dtype, size_out, complex=complex]:
        constrained[not complex, "reduce_min() is only available for non-complex numbers"]()

        return Number[dtype, size_out, complex=complex](rebind[Number[dtype, size_out, complex=complex].Value](self.value.reduce_min[size_out]()))

    @always_inline
    fn reduce_add[size_out: Int = 1](self) -> Number[dtype, size_out, complex=complex]:
        @always_inline
        @parameter
        fn reduce_add_body[
            dtype: DType, width: Int
        ](v1: Number[dtype, width, complex=complex], v2: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return v1 + v2

        return self.reduce[reduce_add_body, size_out]()

    @always_inline
    fn reduce_mul[size_out: Int = 1](self) -> Number[dtype, size_out, complex=complex]:
        @always_inline
        @parameter
        fn reduce_mul_body[
            dtype: DType, width: Int
        ](v1: Number[dtype, width, complex=complex], v2: Number[dtype, width, complex=complex]) -> Number[dtype, width, complex=complex]:
            return v1 * v2

        return self.reduce[reduce_mul_body, size_out]()

    @always_inline
    fn reduce_and[size_out: Int = 1](self) -> Number[dtype, size_out, complex=complex]:
        constrained[not complex, "reduce_and() is only available for non-complex numbers"]()

        return Number[dtype, size_out, complex=complex](rebind[Number[dtype, size_out, complex=complex].Value](self.value.reduce_and[size_out]()))

    @always_inline
    fn reduce_or[size_out: Int = 1](self) -> Number[dtype, size_out, complex=complex]:
        constrained[not complex, "reduce_or() is only available for non-complex numbers"]()

        return Number[dtype, size_out, complex=complex](rebind[Number[dtype, size_out, complex=complex].Value](self.value.reduce_or[size_out]()))

    @always_inline
    fn reduce_bit_count(self) -> Int:
        return self.value.reduce_bit_count()

    @always_inline
    fn select[
        dtype_out: DType, complex_out: Bool, //
    ](self, true_case: Number[dtype_out, width, complex=complex_out], false_case: Number[dtype_out, width, complex=complex_out]) -> Number[
        dtype_out, width, complex=complex_out
    ]:
        constrained[dtype == DType.bool and not complex, "select() is only available for bool, non-complex numbers"]()

        var value = rebind[SIMD[DType.bool, width]](self.value)

        @parameter
        if complex_out:
            return Number[dtype_out, width, complex=complex_out](
                real=value.select(true_case=true_case.real(), false_case=false_case.real()),
                imaginary=value.select(true_case=true_case.imaginary(), false_case=false_case.imaginary()),
            )
        else:
            return Number[dtype_out, width, complex=complex_out](
                rebind[Number[dtype_out, width, complex=complex_out].Value](
                    value.select(true_case=rebind[SIMD[dtype_out, width]](true_case.value), false_case=rebind[SIMD[dtype_out, width]](false_case.value))
                )
            )

    @always_inline
    fn rotate_left[shift: Int](self) -> Self:
        return Self(self.value.rotate_left[2 * shift if complex else shift]())

    @always_inline
    fn rotate_right[shift: Int](self) -> Self:
        return Self(self.value.rotate_right[2 * shift if complex else shift]())

    @always_inline
    fn shift_left[shift: Int](self) -> Self:
        return Self(self.value.shift_left[2 * shift if complex else shift]())

    @always_inline
    fn shift_right[shift: Int](self) -> Self:
        return Self(self.value.shift_right[2 * shift if complex else shift]())

    fn reversed(self) -> Self:
        @parameter
        if complex:
            return Self(real=self.real().reversed(), imaginary=self.imaginary().reversed())
        else:
            return Self(self.value.reversed())

    @always_inline
    fn squared_norm(self) -> Number[dtype, width, complex=False]:
        constrained[complex, "squared_norm() is only available for complex numbers"]()

        return self.real() * self.real() + self.imaginary() * self.imaginary()

    @always_inline
    fn norm(self) -> Number[dtype, width, complex=False]:
        constrained[complex, "norm() is only available for complex numbers"]()

        return sqrt(self.squared_norm().value)

    @always_inline
    fn phase(self) -> Number[dtype, width, complex=False]:
        constrained[complex, "phase() is only available for complex numbers"]()

        return atan2(self.imaginary(), self.real())

    @always_inline
    fn log(self) -> Self:
        @parameter
        if complex:
            return Self(real=log(self.norm().value), imaginary=self.phase().value)
        else:
            return Self(log(self.value))

    #
    # Type Conversion
    #
    @always_inline
    fn cast[new_dtype: DType](self) -> Number[new_dtype, width, complex=complex]:
        @parameter
        if new_dtype == dtype:
            return rebind[Number[new_dtype, width, complex=complex]](self)
        else:
            return Number[new_dtype, width, complex=complex](self.value.cast[new_dtype]())

    @always_inline
    fn as_complex[new_dtype: DType = dtype](self) -> Number[new_dtype, width, complex=True]:
        @parameter
        if complex and new_dtype == dtype:
            return rebind[Number[new_dtype, width, complex=True]](self)
        elif complex:
            return rebind[Number[new_dtype, width, complex=True]](self.cast[new_dtype]())
        else:
            return Number[new_dtype, width, complex=True](real=rebind[Number[new_dtype, width, complex=False].Value](self.value.cast[new_dtype]()), imaginary=0)

    #
    # Stringable & Writable
    #
    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
        @parameter
        if complex:

            @parameter
            if width > 1:
                writer.write("[")

            var real = self.real()
            var imaginary = self.imaginary()

            @parameter
            for i in range(width):
                if i > 0:
                    writer.write(", ")

                var real = round(real[i], ndigits=print_precision)
                var imaginary = round(imaginary[i], ndigits=print_precision)
                real = abs(real) if isclose(real, 0, atol=10**-print_precision) else real
                imaginary = abs(imaginary) if isclose(imaginary, 0, atol=10**-print_precision) else imaginary

                writer.write("(", real, ", ", imaginary, "i)")

            @parameter
            if width > 1:
                writer.write("]")
        else:
            var rounded = round(self.value, ndigits=print_precision)

            if isclose(rounded, 0, atol=10**-print_precision):
                writer.write(abs(rounded))
            else:
                writer.write(rounded)
