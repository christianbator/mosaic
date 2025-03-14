#
# number.mojo
# mosaic
#
# Created by Christian Bator on 03/04/2025
#

from math import sqrt, Ceilable, CeilDivable, Floorable, Truncable
from complex import ComplexSIMD

#
# ScalarNumber
#
alias ScalarNumber = Number[width = 1]

#
# Number
#
@register_passable("trivial")
struct Number[dtype: DType, complex: Bool, width: Int](
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
    Writable
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
    fn __init__[other_dtype: DType, //](out self, value: Number[other_dtype, complex, width]):
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
    fn __init__[T: Floatable](out self: ScalarNumber[DType.float64, complex], value: T):
        @parameter
        if complex:
            self.value = ScalarNumber[DType.float64, complex].Value(value.__float__(), 0.0)
        else:
            self.value = value.__float__()

    @always_inline
    fn __init__[T: FloatableRaising](out self: ScalarNumber[DType.float64, complex], value: T) raises:
        @parameter
        if complex:
            self.value = ScalarNumber[DType.float64, complex].Value(value.__float__(), 0.0)
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
    fn __init__(out self: Number[DType.bool, False, width], value: Bool, /):
        self.value = value

    @always_inline
    @implicit
    fn __init__(out self, value: ScalarNumber[dtype, complex], /):
        @parameter
        if complex:
            self.value = rebind[Self.Value](
                SIMD[dtype, width](value.value[0]).interleave(SIMD[dtype, width](value.value[1]))
            )
        else:
            self.value = value.value[0]

    @always_inline
    @implicit
    fn __init__(out self, *elems: Scalar[dtype]):
        # TODO: Make this a compile-time check when possible
        debug_assert(width == len(elems), "mismatch in the number of elements in the Number variadic constructor")

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

    #
    # TODO: Remove constrained and specify the type of self instead for all constrained methods below?
    #

    @always_inline
    fn __init__(out self, real: SIMD[dtype, width], imaginary: SIMD[dtype, width]):
        constrained[complex, "__init__(real, imaginary) is only available for complex numbers"]()

        self.value = rebind[Self.Value](
            real.interleave(imaginary)
        )

    @staticmethod
    fn from_bits[int_type: DType, //](value: SIMD[int_type, Self.Value.size]) -> Number[dtype, complex, width]:
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
    fn __getitem__(self, index: Int) -> ScalarNumber[dtype, complex]:
        @parameter 
        if complex:
            return ScalarNumber[dtype, complex](self.value[index], self.value[index + 1])
        else:
            return ScalarNumber[dtype, complex](self.value[index])
    
    @always_inline
    fn __setitem__(mut self, index: Int, value: ScalarNumber[dtype, complex]):
        @parameter
        if complex:
            self.value[index] = value.value[0]
            self.value[index + 1] = value.value[1]
        else:
            self.value[index] = value.value[0]

    fn __contains__(self, value: ScalarNumber[dtype, complex]) -> Bool:
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
        
        return rebind[SIMD[dtype, width]](
            self.value.deinterleave()[0]
        )
    
    @always_inline
    fn imaginary(self) -> SIMD[dtype, width]:
        constrained[complex, "imaginary() is only available for complex numbers"]()

        return rebind[SIMD[dtype, width]](
            self.value.deinterleave()[1]
        )

    #
    # Operators
    #
    @always_inline
    fn __pos__(self) -> Self:
        return self

    @always_inline
    fn __neg__(self) -> Self:
        return Self(-self.value)

    @always_inline
    fn __add__(self, rhs: Self) -> Self:
        return Self(self.value + rhs.value)

    @always_inline
    fn __sub__(self, rhs: Self) -> Self:
        return Self(self.value - rhs.value)

    @always_inline
    fn __mul__(self, rhs: Self) -> Self:
        @parameter
        if complex:
            return Self(
                real = self.real() * rhs.real() - self.imaginary() * rhs.imaginary(),
                imaginary = self.real() * rhs.imaginary() + self.imaginary() * rhs.real()
            )
        else:
            return Self(self.value * rhs.value)
    
    @always_inline
    fn __truediv__(self: Self, rhs: Self) -> Self:
        @parameter
        if complex:
            var denominator = rhs.squared_norm()
        
            return Self(
                real = (self.real() * rhs.real() + self.imaginary() * rhs.imaginary()) / denominator,
                imaginary = (self.imaginary() * rhs.real() - self.real() * rhs.imaginary()) / denominator
            )
        else:
            return Self(self.value / rhs.value)
    
    @always_inline
    fn __floordiv__(self: Self, rhs: Self) -> Self:
        constrained[not complex, "__floordiv__() is only available for non-complex numbers"]()

        return Self(self.value // rhs.value)
    
    @always_inline
    fn __mod__(self: Self, rhs: Self) -> Self:
        constrained[not complex, "__mod__() is only available for non-complex numbers"]()

        return Self(self.value % rhs.value)
    
    @always_inline
    fn __pow__(self: Self, rhs: Self) -> Self:
        constrained[not complex, "__pow__() is only available for non-complex numbers"]()

        return Self(self.value ** rhs.value)
    
    @always_inline
    fn __and__(self, rhs: Self) -> Self:
        constrained[not complex, "__and__() is only available for non-complex numbers"]()

        return Self(self.value & rhs.value)

    @always_inline
    fn __or__(self, rhs: Self) -> Self:
        constrained[not complex, "__or__() is only available for non-complex numbers"]()

        return Self(self.value | rhs.value)

    @always_inline
    fn __xor__(self, rhs: Self) -> Self:
        constrained[not complex, "__xor__() is only available for non-complex numbers"]()

        return Self(self.value ^ rhs.value)

    @always_inline
    fn __lshift__(self, rhs: Self) -> Self:
        constrained[not complex, "__lshift__() is only available for non-complex numbers"]()

        return Self(self.value << rhs.value)

    @always_inline
    fn __rshift__(self, rhs: Self) -> Self:
        constrained[not complex, "__rshift__() is only available for non-complex numbers"]()

        return Self(self.value >> rhs.value)

    @always_inline
    fn __lt__(self, rhs: Self) -> SIMD[DType.bool, width]:
        constrained[not complex, "__lt__() is only available for non-complex numbers"]()
        
        return rebind[SIMD[DType.bool, width]](
            self.value < rhs.value
        )
    
    @always_inline
    fn __le__(self, rhs: Self) -> SIMD[DType.bool, width]:
        constrained[not complex, "__le__() is only available for non-complex numbers"]()
        
        return rebind[SIMD[DType.bool, width]](
            self.value <= rhs.value
        )

    @always_inline
    fn __eq__(self, rhs: Self) -> SIMD[DType.bool, width]:
        @parameter
        if complex:
            var result = SIMD[DType.bool, width]()

            @parameter
            for i in range(0, 2 * width, 2):
                result[i // 2] = (self.value[i] == rhs.value[i] and self.value[i + 1] == rhs.value[i + 1])
            
            return result
        else:
            return rebind[SIMD[DType.bool, width]](
                self.value == rhs.value
            )

    @always_inline
    fn __ne__(self, rhs: Self) -> SIMD[DType.bool, width]:
        return not(self == rhs)

    @always_inline
    fn __gt__(self, rhs: Self) -> SIMD[DType.bool, width]:
        constrained[not complex, "__gt__() is only available for non-complex numbers"]()
        
        return rebind[SIMD[DType.bool, width]](
            self.value > rhs.value
        )

    @always_inline
    fn __ge__(self, rhs: Self) -> SIMD[DType.bool, width]:
        constrained[not complex, "__ge__() is only available for non-complex numbers"]()

        return rebind[SIMD[DType.bool, width]](
            self.value >= rhs.value
        )

    @always_inline
    fn __invert__(self) -> Self:
        constrained[not complex, "__invert__() is only available for non-complex numbers"]()

        return Self(self.value.__invert__())

    #
    # Reverse Operators
    #
    @always_inline
    fn __radd__(self, lhs: Self) -> Self:
        return lhs + self

    @always_inline
    fn __rsub__(self, lhs: Self) -> Self:
        return lhs - self

    @always_inline
    fn __rmul__(self, lhs: Self) -> Self:
        return lhs * self
    
    @always_inline
    fn __rtruediv__(self, lhs: Self) -> Self:
        return lhs / self
    
    @always_inline
    fn __rfloordiv__(self, lhs: Self) -> Self:
        return lhs // lhs
    
    @always_inline
    fn __rmod__(self, lhs: Self) -> Self:
        return lhs % self
    
    @always_inline
    fn __rpow__(self, lhs: Self) -> Self:
        return lhs ** self

    @always_inline
    fn __rand__(self, lhs: Self) -> Self:
        return lhs & self

    @always_inline
    fn __ror__(self, lhs: Self) -> Self:
        return lhs | self

    @always_inline
    fn __rxor__(self, lhs: Self) -> Self:
        return lhs ^ self

    @always_inline
    fn __rlshift__(self, lhs: Self) -> Self:
        return lhs << self

    @always_inline
    fn __rrshift__(self, lhs: Self) -> Self:
        return lhs >> self

    #
    # In-place Operators
    #
    @always_inline
    fn __iadd__(mut self, rhs: Self):
        self = self + rhs
    
    @always_inline
    fn __isub__(mut self, rhs: Self):
        self = self - rhs.value

    @always_inline
    fn __imul__(mut self, rhs: Self):
        self = self * rhs
    
    @always_inline
    fn __itruediv__(mut self, rhs: Self):
        self = self / rhs
    
    @always_inline
    fn __ifloordiv__(mut self, rhs: Self):
        self = self // rhs
    
    @always_inline
    fn __imod__(mut self, rhs: Self):
        self = self % rhs
    
    @always_inline
    fn __ipow__(mut self, rhs: Self):
        self = self ** rhs
    
    @always_inline
    fn __iand__(mut self, rhs: Self):
        self = self & rhs

    @always_inline
    fn __ior__(mut self, rhs: Self):
        self = self | rhs

    @always_inline
    fn __ixor__(mut self, rhs: Self):
        self = self ^ rhs

    @always_inline
    fn __ilshift__(mut self, rhs: Self):
        self = self << rhs

    @always_inline
    fn __irshift__(mut self, rhs: Self):
        self = self >> rhs
   
    #
    # Type Conversion
    #
    @always_inline
    fn cast[new_dtype: DType](self) -> Number[new_dtype, complex, width]:
        return Number[new_dtype, complex, width](
            self.value.cast[new_dtype]()
        )

    #
    # Methods
    #

    #
    # TODO: Wrap other SIMD method implementations
    #

    @always_inline
    fn squared_norm(self: Self) -> SIMD[dtype, width]:
        constrained[complex, "squared_norm() is only available for complex numbers"]()

        return self.real() * self.real() + self.imaginary() * self.imaginary()

    @always_inline
    fn norm(self: Self) -> SIMD[dtype, width]:
        constrained[complex, "norm() is only available for complex numbers"]()

        return sqrt(self.squared_norm())
    
    @always_inline
    fn fma(self, multiplier: Self, accumulator: Self) -> Self:
        @parameter
        if complex:
            var result = ComplexSIMD(re = self.real(), im = self.imaginary()).fma(
                b = ComplexSIMD(re = multiplier.real(), im = multiplier.imaginary()),
                c = ComplexSIMD(re = accumulator.real(), im = accumulator.imaginary())
            )

            return Self(real = result.re, imaginary = result.im)
        else:
            return Self(
                self.value.fma(multiplier = multiplier.value, accumulator = accumulator.value)
            )

    #
    # Trait Implementations
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

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @always_inline
    fn __trunc__(self) -> Self:
        constrained[not complex, "__trunc__() is only available for non-complex numbers"]()

        return Self(self.value.__trunc__())
    
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
                
                writer.write("(", real[i], ", ", imaginary[i], "i)")
            
            @parameter
            if width > 1:
                writer.write("]")
        else:
            writer.write(self.value)
