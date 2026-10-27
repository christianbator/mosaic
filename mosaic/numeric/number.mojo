#
# number.mojo
# mosaic
#
# Created by Christian Bator on 03/04/2025
#

import std.math as math
from std.math import sqrt, log, atan2, Ceilable, CeilDivable, Floorable, Truncable, isclose
from std.complex import ComplexSIMD
from std.builtin.dtype import _integral_type_of
from std.sys import is_big_endian
from std.sys.info import size_of
from std.utils import IndexList
from std.hash import Hasher

#
# Type Aliases
#
comptime ScalarNumber = Number[_, width=1, complex=_]

#
# Print Precision
#
comptime print_precision = 3


#
# Number
#
struct Number[dtype: DType, width: Int, *, complex: Bool = False](
    Absable,
    Boolable,
    CeilDivable,
    Ceilable,
    Copyable,
    Floatable,
    Floorable,
    Hashable,
    Indexer,
    Intable,
    Roundable,
    TrivialRegisterPassable,
    Truncable,
    Writable,
):
    #
    # Fields
    #
    comptime Value = SIMD[Self.dtype, 2 * Self.width if Self.complex else Self.width]

    comptime MAX = Self._max()
    comptime MIN = Self(Self.Value.MIN)
    comptime MAX_FINITE = Self(Self.Value.MAX_FINITE)
    comptime MIN_FINITE = Self(Self.Value.MIN_FINITE)

    var value: Self.Value

    #
    # Initialization
    #
    @always_inline
    def __init__(out self):
        self.value = Self.Value(0)

    @always_inline
    @implicit
    def __init__(out self, value: Self.Value):
        self.value = value

    @always_inline
    def __init__[other_dtype: DType, //](out self, value: Number[other_dtype, Self.width, complex=Self.complex]):
        self.value = value.value.cast[Self.dtype]()

    @always_inline
    @implicit
    def __init__(out self, value: UInt):
        comptime if Self.complex:
            self = Self(real=SIMD[Self.dtype, Self.width](value), imaginary=0)
        else:
            self.value = Self.Value(value)

    @always_inline
    @implicit
    def __init__(out self, value: Int):
        comptime if Self.complex:
            self = Self(real=SIMD[Self.dtype, Self.width](value), imaginary=0)
        else:
            self.value = Self.Value(value)

    @always_inline
    def __init__[T: Floatable](out self: ScalarNumber[DType.float64, complex=Self.complex], value: T):
        comptime if Self.complex:
            self.value = ScalarNumber[DType.float64, complex=Self.complex].Value(value.__float__(), 0.0)
        else:
            self.value = value.__float__()

    @always_inline
    def __init__[T: FloatableRaising](out self: ScalarNumber[DType.float64, complex=Self.complex], value: T) raises:
        comptime if Self.complex:
            self.value = ScalarNumber[DType.float64, complex=Self.complex].Value(value.__float__(), 0.0)
        else:
            self.value = value.__float__()

    @always_inline
    @implicit
    def __init__(out self, value: IntLiteral):
        comptime if Self.complex:
            self.value = Self.Value(value, 0)
        else:
            self.value = value

    @always_inline
    @implicit
    def __init__(out self, value: FloatLiteral):
        comptime if Self.complex:
            self.value = Self.Value(value, 0)
        else:
            self.value = value

    @always_inline
    @implicit
    def __init__(out self: Number[DType.bool, Self.width, complex=False], value: Bool, /):
        self.value = value

    @always_inline
    @implicit
    def __init__(out self, value: ScalarNumber[Self.dtype, complex=Self.complex], /):
        comptime if Self.complex:
            self.value = rebind[Self.Value](
                SIMD[Self.dtype, Self.width](value.value[0]).interleave(SIMD[Self.dtype, Self.width](value.value[1]))
            )
        else:
            self.value = value.value[0]

    @always_inline
    @implicit
    def __init__(out self, *elems: Scalar[Self.dtype]):
        # TODO: Make this a compile-time check when possible
        debug_assert(Self.width == len(elems), "Mismatch in the number of elements in the Number variadic constructor")

        self.value = Self.Value(0)

        comptime if Self.complex:
            comptime for i in range(0, 2 * Self.width, 2):
                self.value[i] = elems[i]
        else:
            comptime for i in range(Self.width):
                self.value[i] = elems[i]

    @always_inline
    def __init__(out self, real: SIMD[Self.dtype, Self.width], imaginary: SIMD[Self.dtype, Self.width]):
        comptime assert Self.complex, "__init__(real, imaginary) is only available for complex numbers"

        self.value = rebind[Self.Value](real.interleave(imaginary))

    @always_inline
    @implicit
    def __init__(out self, tuple: Tuple[SIMD[Self.dtype, Self.width], SIMD[Self.dtype, Self.width]]):
        comptime assert Self.complex, "__init__(tuple) is only available for complex numbers"

        self = Self(real=tuple[0], imaginary=tuple[1])

    @staticmethod
    def from_bits[int_type: DType, //](value: SIMD[int_type, Self.Value.length]) -> Number[Self.dtype, Self.width, complex=Self.complex]:
        return Self(Self.Value(from_bits=value))

    @staticmethod
    def _max() -> Self:
        comptime assert not Self.complex, "MAX is only available for non-complex numbers"

        return Self(Self.Value.MAX)

    @staticmethod
    def _min() -> Self:
        comptime assert not Self.complex, "MIN is only available for non-complex numbers"

        return Self(Self.Value.MIN)

    @staticmethod
    def _max_finite() -> Self:
        comptime assert not Self.complex, "MAX_FINITE is only available for non-complex numbers"

        return Self(Self.Value.MAX_FINITE)

    @staticmethod
    def _min_finite() -> Self:
        comptime assert not Self.complex, "MIN_FINITE is only available for non-complex numbers"

        return Self(Self.Value.MIN_FINITE)

    #
    # Access
    #
    @always_inline
    def __getitem__(self, index: Int) -> ScalarNumber[Self.dtype, complex=Self.complex]:
        comptime if Self.complex:
            return ScalarNumber[Self.dtype, complex=Self.complex](real=self.value[index], imaginary=self.value[index + 1])
        else:
            return ScalarNumber[Self.dtype, complex=Self.complex](self.value[index])

    @always_inline
    def __setitem__(mut self, index: Int, value: ScalarNumber[Self.dtype, complex=Self.complex]):
        comptime if Self.complex:
            self.value[index] = value.value[0]
            self.value[index + 1] = value.value[1]
        else:
            self.value[index] = value.value[0]

    def __contains__(self, value: ScalarNumber[Self.dtype, complex=Self.complex]) -> Bool:
        comptime if Self.complex:
            comptime for i in range(Self.width):
                if self[i] == value:
                    return True

            return False
        else:
            return self.value.__contains__(value.value[0])

    @always_inline
    def real(self) -> SIMD[Self.dtype, Self.width]:
        comptime assert Self.complex, "real() is only available for complex numbers"

        return rebind[SIMD[Self.dtype, Self.width]](self.value.deinterleave()[0])

    @always_inline
    def imaginary(self) -> SIMD[Self.dtype, Self.width]:
        comptime assert Self.complex, "imaginary() is only available for complex numbers"

        return rebind[SIMD[Self.dtype, Self.width]](self.value.deinterleave()[1])

    #
    # Operators
    #
    @always_inline
    def __neg__(self) -> Self:
        return Self(-self.value)

    @always_inline
    def __add__(self, rhs: Self) -> Self:
        return Self(self.value + rhs.value)

    @always_inline
    def __radd__(self, lhs: Self) -> Self:
        return lhs + self

    @always_inline
    def __iadd__(mut self, rhs: Self):
        self = self + rhs

    @always_inline
    def __sub__(self, rhs: Self) -> Self:
        return Self(self.value - rhs.value)

    @always_inline
    def __rsub__(self, lhs: Self) -> Self:
        return lhs - self

    @always_inline
    def __isub__(mut self, rhs: Self):
        self = self - rhs

    @always_inline
    def __mul__(self, rhs: Self) -> Self:
        comptime if Self.complex:
            return Self(
                real=self.real() * rhs.real() - self.imaginary() * rhs.imaginary(),
                imaginary=self.real() * rhs.imaginary() + self.imaginary() * rhs.real(),
            )
        else:
            return Self(self.value * rhs.value)

    @always_inline
    def __rmul__(self, lhs: Self) -> Self:
        return lhs * self

    @always_inline
    def __imul__(mut self, rhs: Self):
        self = self * rhs

    @always_inline
    def __truediv__(self, rhs: Self) -> Self:
        comptime if Self.complex:
            var denominator = rhs.squared_norm().value

            return Self(
                real=(self.real() * rhs.real() + self.imaginary() * rhs.imaginary()) / denominator,
                imaginary=(self.imaginary() * rhs.real() - self.real() * rhs.imaginary()) / denominator,
            )
        else:
            return Self(self.value / rhs.value)

    @always_inline
    def __rtruediv__(self, lhs: Self) -> Self:
        return lhs / self

    @always_inline
    def __itruediv__(mut self, rhs: Self):
        self = self / rhs

    @always_inline
    def __floordiv__(self, rhs: Self) -> Self:
        comptime assert not Self.complex, "__floordiv__() is only available for non-complex numbers"

        return Self(self.value // rhs.value)

    @always_inline
    def __rfloordiv__(self, lhs: Self) -> Self:
        return lhs // lhs

    @always_inline
    def __ifloordiv__(mut self, rhs: Self):
        self = self // rhs

    @always_inline
    def __mod__(self, rhs: Self) -> Self:
        comptime assert not Self.complex, "__mod__() is only available for non-complex numbers"

        return Self(self.value % rhs.value)

    @always_inline
    def __rmod__(self, lhs: Self) -> Self:
        return lhs % self

    @always_inline
    def __imod__(mut self, rhs: Self):
        self = self % rhs

    @always_inline
    def __pow__(self, rhs: Self) -> Self:
        comptime assert not Self.complex, "__pow__() is only available for non-complex numbers"

        return Self(self.value**rhs.value)

    @always_inline
    def __rpow__(self, lhs: Self) -> Self:
        return lhs**self

    @always_inline
    def __ipow__(mut self, rhs: Self):
        self = self**rhs

    @always_inline
    def __and__(self, rhs: Self) -> Self:
        comptime assert not Self.complex, "__and__() is only available for non-complex numbers"

        return Self(self.value & rhs.value)

    @always_inline
    def __rand__(self, lhs: Self) -> Self:
        return lhs & self

    @always_inline
    def __iand__(mut self, rhs: Self):
        self = self & rhs

    @always_inline
    def __or__(self, rhs: Self) -> Self:
        comptime assert not Self.complex, "__or__() is only available for non-complex numbers"

        return Self(self.value | rhs.value)

    @always_inline
    def __ror__(self, lhs: Self) -> Self:
        return lhs | self

    @always_inline
    def __ior__(mut self, rhs: Self):
        self = self | rhs

    @always_inline
    def __xor__(self, rhs: Self) -> Self:
        comptime assert not Self.complex, "__xor__() is only available for non-complex numbers"

        return Self(self.value ^ rhs.value)

    @always_inline
    def __rxor__(self, lhs: Self) -> Self:
        return lhs ^ self

    @always_inline
    def __ixor__(mut self, rhs: Self):
        self = self ^ rhs

    @always_inline
    def __lshift__(self, rhs: Self) -> Self:
        comptime assert not Self.complex, "__lshift__() is only available for non-complex numbers"

        return Self(self.value << rhs.value)

    @always_inline
    def __rlshift__(self, lhs: Self) -> Self:
        return lhs << self

    @always_inline
    def __ilshift__(mut self, rhs: Self):
        self = self << rhs

    @always_inline
    def __rshift__(self, rhs: Self) -> Self:
        comptime assert not Self.complex, "__rshift__() is only available for non-complex numbers"

        return Self(self.value >> rhs.value)

    @always_inline
    def __rrshift__(self, lhs: Self) -> Self:
        return lhs >> self

    @always_inline
    def __irshift__(mut self, rhs: Self):
        self = self >> rhs

    @always_inline
    def __lt__(self, rhs: Self) -> SIMD[DType.bool, Self.width]:
        comptime assert not Self.complex, "__lt__() is only available for non-complex numbers"

        return rebind[SIMD[DType.bool, Self.width]](self.value.lt(rhs.value))

    @always_inline
    def __le__(self, rhs: Self) -> SIMD[DType.bool, Self.width]:
        comptime assert not Self.complex, "__le__() is only available for non-complex numbers"

        return rebind[SIMD[DType.bool, Self.width]](self.value.le(rhs.value))

    @always_inline
    def __eq__(self, rhs: Self) -> SIMD[DType.bool, Self.width]:
        comptime if Self.complex:
            var result = SIMD[DType.bool, Self.width]()

            comptime for i in range(0, 2 * Self.width, 2):
                result[i // 2] = self.value[i] == rhs.value[i] and self.value[i + 1] == rhs.value[i + 1]

            return result
        else:
            return rebind[SIMD[DType.bool, Self.width]](self.value.eq(rhs.value))

    @always_inline
    def __ne__(self, rhs: Self) -> SIMD[DType.bool, Self.width]:
        comptime if Self.complex:
            return ~self.__eq__(rhs)
        else:
            return rebind[SIMD[DType.bool, Self.width]](self.value.ne(rhs.value))

    @always_inline
    def __gt__(self, rhs: Self) -> SIMD[DType.bool, Self.width]:
        comptime assert not Self.complex, "__gt__() is only available for non-complex numbers"

        return rebind[SIMD[DType.bool, Self.width]](self.value.gt(rhs.value))

    @always_inline
    def __ge__(self, rhs: Self) -> SIMD[DType.bool, Self.width]:
        comptime assert not Self.complex, "__ge__() is only available for non-complex numbers"

        return rebind[SIMD[DType.bool, Self.width]](self.value.ge(rhs.value))

    @always_inline
    def __invert__(self) -> Self:
        comptime assert not Self.complex, "__invert__() is only available for non-complex numbers"

        return Self(self.value.__invert__())

    #
    # Numeric Traits
    #
    @always_inline
    def __abs__(self) -> Self:
        comptime assert not Self.complex, "__abs__() is only available for non-complex numbers, see norm() instead"

        return Self(self.value.__abs__())

    @always_inline
    def __bool__(self) -> Bool:
        comptime assert not Self.complex, "__bool__() is only available for non-complex numbers"

        return self.value.__bool__()

    @always_inline
    def __ceil__(self) -> Self:
        comptime assert not Self.complex, "__ceil__() is only available for non-complex numbers"

        return Self(self.value.__ceil__())

    @always_inline
    def __ceildiv__(self, denominator: Self) -> Self:
        comptime assert not Self.complex, "__ceildiv__() is only available for non-complex numbers"

        return Self(self.value.__ceildiv__(denominator.value))

    @always_inline
    def __float__(self) -> Float64:
        comptime assert not Self.complex, "__float__() is only available for non-complex numbers"

        return self.value.__float__()

    @always_inline
    def __floor__(self) -> Self:
        comptime assert not Self.complex, "__floor__() is only available for non-complex numbers"

        return Self(self.value.__floor__())

    def __hash__[H: Hasher](self, mut hasher: H):
        return self.value.__hash__(hasher)

    @always_inline
    def __int__(self) -> Int:
        comptime assert not Self.complex, "__int__() is only available for non-complex numbers"

        return self.value.__int__()

    @always_inline
    def __mlir_index__(self) -> __mlir_type.index:
        comptime assert not Self.complex, "__mlir_index__() is only available for non-complex numbers"

        return self.__int__().__mlir_index__()

    @always_inline
    def __round__(self) -> Self:
        comptime assert not Self.complex, "__round__() is only available for non-complex numbers"

        return Self(self.value.__round__())

    @always_inline
    def __round__(self, ndigits: Int) -> Self:
        comptime assert not Self.complex, "__round__(ndigits) is only available for non-complex numbers"

        return Self(self.value.__round__(ndigits))

    @always_inline
    def __trunc__(self) -> Self:
        comptime assert not Self.complex, "__trunc__() is only available for non-complex numbers"

        return Self(self.value.__trunc__())

    #
    # Numeric Methods
    #
    @always_inline
    def to_bits[int_dtype: DType = _integral_type_of[Self.dtype]()](self) -> Number[int_dtype, Self.width, complex=Self.complex]:
        return Number[int_dtype, Self.width, complex=Self.complex](self.value.to_bits())

    # @staticmethod
    # def from_bytes[*, big_endian: Bool = is_big_endian()](bytes: Array[Byte, size_of[SIMD[Self.dtype, Self.width]]()]) -> Self:
    #     comptime assert Self.width == 1 and not Self.complex, "from_bytes() is only available for scalar, non-complex numbers"

    #     var result: Scalar[Self.dtype] = Scalar[Self.dtype].from_bytes(bytes)

    #     return Self(result)

    # def as_bytes[*, big_endian: Bool = is_big_endian()](self) -> Array[Byte, size_of[SIMD[Self.dtype, Self.width]]()]:
    #     comptime assert Self.width == 1 and not Self.complex, "as_bytes() is only available for scalar, non-complex numbers"

    #     return self.value.as_bytes()

    def clamp(
        self, lower_bound: Number[Self.dtype, Self.width, complex=False], upper_bound: Number[Self.dtype, Self.width, complex=False]
    ) -> Self:
        comptime assert not Self.complex, "clamp() is only available for non-complex numbers"

        return Self(self.value.clamp(lower_bound=rebind[Self.Value](lower_bound.value), upper_bound=rebind[Self.Value](upper_bound.value)))

    @always_inline
    def fma(self, multiplier: Self, accumulator: Self) -> Self:
        comptime if Self.complex:
            var result = ComplexSIMD(re=self.real(), im=self.imaginary()).fma(
                b=ComplexSIMD(re=multiplier.real(), im=multiplier.imaginary()),
                c=ComplexSIMD(re=accumulator.real(), im=accumulator.imaginary()),
            )

            return Self(real=result.re, imaginary=result.im)
        else:
            return Self(self.value.fma(multiplier=multiplier.value, accumulator=accumulator.value))

    @always_inline
    def shuffle[*mask: SIMDLength](self) -> Self:
        return self.shuffle[*mask](self)

    @always_inline
    def shuffle[*mask: SIMDLength](self, other: Self) -> Self:
        comptime if Self.complex:
            return Self(real=self.real().shuffle[*mask](other.real()), imaginary=self.imaginary().shuffle[*mask](other.imaginary()))
        else:
            return Self(self.value.shuffle[*mask](other.value))

    @always_inline
    def shuffle[mask: IndexList[Self.width, element_type=_]](self) -> Self:
        return self.shuffle[mask=mask](self)

    @always_inline
    def shuffle[mask: IndexList[Self.width, element_type=_]](self, other: Self) -> Self:
        comptime if Self.complex:
            return Self(real=self.real().shuffle[mask=mask](other.real()), imaginary=self.imaginary().shuffle[mask=mask](other.imaginary()))
        else:
            return Self(
                rebind[Self.Value](
                    rebind[SIMD[Self.dtype, Self.width]](self.value).shuffle[mask=mask](rebind[SIMD[Self.dtype, Self.width]](other.value))
                )
            )

    @always_inline
    def slice[output_width: Int, /, *, offset: Int = 0](self) -> Number[Self.dtype, output_width, complex=Self.complex]:
        return Number[Self.dtype, output_width, complex=Self.complex](
            rebind[Number[Self.dtype, output_width, complex=Self.complex].Value](
                self.value.slice[2 * output_width if Self.complex else output_width, offset=2 * offset if Self.complex else offset]()
            )
        )

    @always_inline
    def insert[*, offset: Int = 0](self, value: Number[Self.dtype, _, complex=Self.complex]) -> Self:
        return Self(self.value.insert[offset=2 * offset if Self.complex else offset](value.value))

    @always_inline
    def join(self, other: Self, out result: Number[Self.dtype, 2 * Self.width, complex=Self.complex]):
        result = Number[Self.dtype, 2 * Self.width, complex=Self.complex](
            rebind[Number[Self.dtype, 2 * Self.width, complex=Self.complex].Value](self.value.join(other.value))
        )

    @always_inline
    def interleave(self, other: Self) -> Number[Self.dtype, 2 * Self.width, complex=Self.complex]:
        comptime if Self.complex:
            return Number[Self.dtype, 2 * Self.width, complex=Self.complex](
                real=self.real().interleave(other.real()), imaginary=self.imaginary().interleave(other.imaginary())
            )
        else:
            return Number[Self.dtype, 2 * Self.width, complex=Self.complex](
                rebind[Number[Self.dtype, 2 * Self.width, complex=Self.complex].Value](self.value.interleave(other.value))
            )

    @always_inline
    def split(
        self,
    ) -> Tuple[Number[Self.dtype, Self.width // 2, complex=Self.complex], Number[Self.dtype, Self.width // 2, complex=Self.complex]]:
        var split = self.value.split()

        return (
            Number[Self.dtype, Self.width // 2, complex=Self.complex](
                rebind[Number[Self.dtype, Self.width // 2, complex=Self.complex].Value](split[0])
            ),
            Number[Self.dtype, Self.width // 2, complex=Self.complex](
                rebind[Number[Self.dtype, Self.width // 2, complex=Self.complex].Value](split[1])
            ),
        )

    @always_inline
    def deinterleave(
        self,
    ) -> Tuple[Number[Self.dtype, Self.width // 2, complex=Self.complex], Number[Self.dtype, Self.width // 2, complex=Self.complex]]:
        comptime if Self.complex:
            var real = self.real().deinterleave()
            var imaginary = self.imaginary().deinterleave()

            return (
                Number[Self.dtype, Self.width // 2, complex=Self.complex](
                    real=rebind[SIMD[Self.dtype, Self.width // 2]](real[0]),
                    imaginary=rebind[SIMD[Self.dtype, Self.width // 2]](imaginary[0]),
                ),
                Number[Self.dtype, Self.width // 2, complex=Self.complex](
                    real=rebind[SIMD[Self.dtype, Self.width // 2]](real[1]),
                    imaginary=rebind[SIMD[Self.dtype, Self.width // 2]](imaginary[1]),
                ),
            )
        else:
            var deinterleaved = self.value.deinterleave()

            return (
                Number[Self.dtype, Self.width // 2, complex=Self.complex](
                    rebind[Number[Self.dtype, Self.width // 2, complex=Self.complex].Value](deinterleaved[0])
                ),
                Number[Self.dtype, Self.width // 2, complex=Self.complex](
                    rebind[Number[Self.dtype, Self.width // 2, complex=Self.complex].Value](deinterleaved[1])
                ),
            )

    @always_inline
    def reduce[
        func: def[dtype: DType, width: Int](
            Number[dtype, width, complex=Self.complex], Number[dtype, width, complex=Self.complex]
        ) thin -> Number[dtype, width, complex=Self.complex],
        size_out: Int = 1,
    ](self) -> Number[Self.dtype, size_out, complex=Self.complex]:
        comptime assert size_out <= Self.width, "Reduce must specify width less than original number width"

        comptime if Self.width == size_out:
            return rebind[Number[Self.dtype, size_out, complex=Self.complex]](self)
        else:
            var lhs: Number[Self.dtype, Self.width // 2, complex=Self.complex]
            var rhs: Number[Self.dtype, Self.width // 2, complex=Self.complex]
            lhs, rhs = self.split()

            return func(lhs, rhs).reduce[func, size_out]()

    @always_inline
    def reduce_max[size_out: Int = 1](self) -> Number[Self.dtype, size_out, complex=Self.complex]:
        comptime assert not Self.complex, "reduce_max() is only available for non-complex numbers"

        return Number[Self.dtype, size_out, complex=Self.complex](
            rebind[Number[Self.dtype, size_out, complex=Self.complex].Value](self.value.reduce_max[size_out]())
        )

    @always_inline
    def reduce_min[size_out: Int = 1](self) -> Number[Self.dtype, size_out, complex=Self.complex]:
        comptime assert not Self.complex, "reduce_min() is only available for non-complex numbers"

        return Number[Self.dtype, size_out, complex=Self.complex](
            rebind[Number[Self.dtype, size_out, complex=Self.complex].Value](self.value.reduce_min[size_out]())
        )

    @always_inline
    def reduce_add[size_out: Int = 1](self) -> Number[Self.dtype, size_out, complex=Self.complex]:
        @always_inline
        def reduce_add_body[
            dtype: DType, width: Int
        ](v1: Number[dtype, width, complex=Self.complex], v2: Number[dtype, width, complex=Self.complex]) -> Number[
            dtype, width, complex=Self.complex
        ]:
            return v1 + v2

        return self.reduce[reduce_add_body, size_out]()

    @always_inline
    def reduce_mul[size_out: Int = 1](self) -> Number[Self.dtype, size_out, complex=Self.complex]:
        @always_inline
        def reduce_mul_body[
            dtype: DType, width: Int
        ](v1: Number[dtype, width, complex=Self.complex], v2: Number[dtype, width, complex=Self.complex]) -> Number[
            dtype, width, complex=Self.complex
        ]:
            return v1 * v2

        return self.reduce[reduce_mul_body, size_out]()

    @always_inline
    def reduce_and[size_out: Int = 1](self) -> Number[Self.dtype, size_out, complex=Self.complex]:
        comptime assert not Self.complex, "reduce_and() is only available for non-complex numbers"

        return Number[Self.dtype, size_out, complex=Self.complex](
            rebind[Number[Self.dtype, size_out, complex=Self.complex].Value](self.value.reduce_and[size_out]())
        )

    @always_inline
    def reduce_or[size_out: Int = 1](self) -> Number[Self.dtype, size_out, complex=Self.complex]:
        comptime assert not Self.complex, "reduce_or() is only available for non-complex numbers"

        return Number[Self.dtype, size_out, complex=Self.complex](
            rebind[Number[Self.dtype, size_out, complex=Self.complex].Value](self.value.reduce_or[size_out]())
        )

    @always_inline
    def reduce_bit_count(self) -> Int:
        return self.value.reduce_bit_count()

    @always_inline
    def select[
        dtype_out: DType, complex_out: Bool, //
    ](
        self, true_case: Number[dtype_out, Self.width, complex=complex_out], false_case: Number[dtype_out, Self.width, complex=complex_out]
    ) -> Number[dtype_out, Self.width, complex=complex_out]:
        comptime assert Self.dtype == DType.bool and not Self.complex, "select() is only available for bool, non-complex numbers"

        var value = rebind[SIMD[DType.bool, Self.width]](self.value)

        comptime if complex_out:
            return Number[dtype_out, Self.width, complex=complex_out](
                real=value.select(true_case=true_case.real(), false_case=false_case.real()),
                imaginary=value.select(true_case=true_case.imaginary(), false_case=false_case.imaginary()),
            )
        else:
            return Number[dtype_out, Self.width, complex=complex_out](
                rebind[Number[dtype_out, Self.width, complex=complex_out].Value](
                    value.select(
                        true_case=rebind[SIMD[dtype_out, Self.width]](true_case.value),
                        false_case=rebind[SIMD[dtype_out, Self.width]](false_case.value),
                    )
                )
            )

    @always_inline
    def rotate_left[shift: Int](self) -> Self:
        return Self(self.value.rotate_left[2 * shift if Self.complex else shift]())

    @always_inline
    def rotate_right[shift: Int](self) -> Self:
        return Self(self.value.rotate_right[2 * shift if Self.complex else shift]())

    @always_inline
    def shift_left[shift: Int](self) -> Self:
        return Self(self.value.shift_left[2 * shift if Self.complex else shift]())

    @always_inline
    def shift_right[shift: Int](self) -> Self:
        return Self(self.value.shift_right[2 * shift if Self.complex else shift]())

    def reversed(self) -> Self:
        comptime if Self.complex:
            return Self(real=self.real().reversed(), imaginary=self.imaginary().reversed())
        else:
            return Self(self.value.reversed())

    @always_inline
    def squared_norm(self) -> Number[Self.dtype, Self.width, complex=False]:
        comptime assert Self.complex, "squared_norm() is only available for complex numbers"

        return self.real() * self.real() + self.imaginary() * self.imaginary()

    @always_inline
    def norm(self) -> Number[Self.dtype, Self.width, complex=False]:
        comptime assert Self.complex, "norm() is only available for complex numbers"

        return sqrt(self.squared_norm().value)

    @always_inline
    def phase(self) -> Number[Self.dtype, Self.width, complex=False] where Self.dtype.is_floating_point():
        comptime assert Self.complex, "phase() is only available for complex numbers"

        return atan2(self.imaginary(), self.real())

    @always_inline
    def log(self) -> Self where Self.dtype.is_floating_point():
        comptime if Self.complex:
            return Self(real=math.log(self.norm().value), imaginary=self.phase().value)
        else:
            return Self(math.log(self.value))

    #
    # Type Conversion
    #
    @always_inline
    def cast[new_dtype: DType](self) -> Number[new_dtype, Self.width, complex=Self.complex]:
        comptime if new_dtype == Self.dtype:
            return rebind[Number[new_dtype, Self.width, complex=Self.complex]](self)
        else:
            return Number[new_dtype, Self.width, complex=Self.complex](self.value.cast[new_dtype]())

    @always_inline
    def as_complex[new_dtype: DType = Self.dtype](self) -> Number[new_dtype, Self.width, complex=True]:
        comptime if Self.complex and new_dtype == Self.dtype:
            return rebind[Number[new_dtype, Self.width, complex=True]](self)
        elif Self.complex:
            return rebind[Number[new_dtype, Self.width, complex=True]](self.cast[new_dtype]())
        else:
            return Number[new_dtype, Self.width, complex=True](
                real=rebind[Number[new_dtype, Self.width, complex=False].Value](self.value.cast[new_dtype]()), imaginary=0
            )

    #
    # Writable
    #
    @no_inline
    def write_to[W: Writer](self, mut writer: W):
        comptime if Self.complex:
            comptime if Self.width > 1:
                writer.write("[")

            var real = self.real()
            var imaginary = self.imaginary()

            comptime for i in range(Self.width):
                if i > 0:
                    writer.write(", ")

                var real = round(real[i], ndigits=print_precision)
                var imaginary = round(imaginary[i], ndigits=print_precision)
                real = abs(real) if isclose(real, 0, atol=10**-print_precision) else real
                imaginary = abs(imaginary) if isclose(imaginary, 0, atol=10**-print_precision) else imaginary

                writer.write("(", real, ", ", imaginary, "i)")

            comptime if Self.width > 1:
                writer.write("]")
        else:
            comptime if Self.dtype.is_floating_point():
                var rounded = round(self.value, ndigits=print_precision)

                if isclose(rounded, 0, atol=10**-print_precision):
                    writer.write(abs(rounded))
                else:
                    writer.write(rounded)
            else:
                writer.write(self.value)
