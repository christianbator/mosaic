#
# fft.mojo
# mosaic
#
# Created by Christian Bator on 04/01/2025
#

from math import pi, cos, sin, sqrt
from bit import is_power_of_two, bit_width
from memory import memset_zero

from mosaic.utility import fatal_error, print_list


#
# FFT
#
fn fft[dtype: DType, depth: Int, //](matrix: Matrix[dtype, depth, complex=False]) -> Matrix[DType.float64, depth, complex=True]:
    var result = Matrix[DType.float64, depth, complex=True](rows=matrix.rows(), cols=matrix.cols())

    # Rows
    _mixed_radix_fft(matrix, result)

    # Cols
    # result.transpose()
    # _mixed_radix_fft(result)
    # result.transpose()

    return result^


#
# Mixed-Radix FFT
#
fn _mixed_radix_fft[dtype: DType, depth: Int, //](matrix: Matrix[dtype, depth], mut result: Matrix[DType.float64, depth, complex=True]):
    var plan = _FactorPlan(matrix.cols())
    print("\nActual:  ", end="")
    print_list(plan.actual)
    print("Sofar:   ", end="")
    print_list(plan.sofar)
    print("Remain:  ", end="")
    print_list(plan.remain)
    print()

    _permute(matrix, plan, result)
    print("Reordered: ", end="")
    # print(result)
    print()

    _dispatch[twiddle=False](plan.actual[1], plan.sofar[1], plan.remain[1], result)

    for i in range(2, len(plan.actual)):
        _dispatch[twiddle=True](plan.actual[i], plan.sofar[i], plan.remain[i], result)


#
# FactorPlan
#
struct _FactorPlan:
    var actual: List[Int]
    var sofar: List[Int]
    var remain: List[Int]

    fn __init__(out self, N: Int):
        self.actual = List[Int](0) + Self._factor(N)
        self.sofar = List[Int](0, 1)
        self.remain = List[Int](N, N // self.actual[1])

        for i in range(2, len(self.actual)):
            self.sofar.append(self.sofar[i - 1] * self.actual[i - 1])
            self.remain.append(self.remain[i - 1] // self.actual[i])

    @staticmethod
    fn _factor(owned N: Int) -> List[Int]:
        var factors = List[Int]()

        if N == 1:
            factors.append(1)
            return factors

        alias radices = InlineArray[Int, 7](2, 3, 4, 5, 7, 8, 10)

        var i = len(radices) - 1

        # Factor by supported radices in decreasing order
        while (N > 1) and (i >= 0):
            if (N % radices[i]) == 0:
                N //= radices[i]
                factors.append(radices[i])
            else:
                i -= 1

        # Substitute factors 2 * 8 with 4 * 4
        if factors[-1] == 2:
            i = len(factors) - 1
            while (i >= 0) and (factors[i] != 8):
                i -= 1

            if i >= 0:
                factors[-1] = 4
                factors[i] = 4

        # Try ascending odd factors
        for k in range(3, sqrt(N) + 1, 2):
            while (N % k) == 0:
                N //= k
                factors.append(k)

        # N is prime
        if N > 1:
            factors.append(N)

        return factors


#
# Permute
#
fn _permute[
    dtype: DType, depth: Int, //
](matrix: Matrix[dtype, depth, complex=False], plan: _FactorPlan, mut result: Matrix[DType.float64, depth, complex=True]):
    try:
        var N = matrix.cols()

        var count = NumericArray[DType.index](count=len(plan.actual))

        var k = 0
        for i in range(N - 1):
            result[0, i, 0] = matrix[0, k, 0].as_complex[DType.float64]()

            var j = 1
            k += plan.remain[j]
            count[1] += 1
            while count[j] >= plan.actual[j]:
                count[j] = 0
                k = k - plan.remain[j - 1] + plan.remain[j + 1]
                j += 1
                count[j] += 1

        result[0, N - 1, 0] = matrix[0, N - 1, 0].as_complex[DType.float64]()

    except error:
        fatal_error(error)


#
# Dispatch
#
fn _dispatch[depth: Int, //, *, twiddle: Bool](radix: Int, sofar_radix: Int, remain_radix: Int, mut result: Matrix[DType.float64, depth, complex=True]):
    try:
        # Radix-specific method dispatch
        var twiddle_factors = NumericArray[DType.float64, complex=True](count=radix)
        var omega = 2.0 * pi / (sofar_radix * radix)
        var base_twiddle_factor: ScalarNumber[DType.float64, complex=True] = (cos(omega), -sin(omega))
        var twiddle_factor: ScalarNumber[DType.float64, complex=True] = (1.0, 0.0)

        for group_offset in range(sofar_radix):

            @parameter
            if twiddle:
                twiddle_factors[0] = (1.0, 0.0)
                twiddle_factors[1] = twiddle_factor

                for i in range(2, radix):
                    twiddle_factors[i] = twiddle_factor * twiddle_factors[i - 1]

                twiddle_factor *= base_twiddle_factor

            if radix == 2:
                _radix_2_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, result)
            elif radix == 3:
                _radix_3_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, result)
            elif radix == 4:
                _radix_4_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, result)
            elif radix == 5:
                _radix_5_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, result)
            elif radix == 7:
                _radix_7_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, result)
            elif radix == 8:
                _radix_8_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, result)
            elif radix == 10:
                _radix_10_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, result)
            else:
                _radix_prime_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, result)

    except error:
        fatal_error(error)


#
# Radix-2
#
fn _radix_2_fft[
    depth: Int, //, twiddle: Bool
](
    radix: Int,
    sofar_radix: Int,
    remain_radix: Int,
    owned group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    mut result: Matrix[DType.float64, depth, complex=True],
):
    try:
        var offset = group_offset
        var indices = InlineArray[Int, 2](fill=0)

        for _ in range(remain_radix):
            for block in range(radix):
                indices[block] = offset
                offset += sofar_radix

            var value_0 = result[0, indices[0], 0]
            var value_1 = result[0, indices[1], 0]

            @parameter
            if twiddle:
                value_1 *= twiddle_factors[1]

            result[0, indices[0], 0] = value_0 + value_1
            result[0, indices[1], 0] = value_0 - value_1

            group_offset = group_offset + sofar_radix * radix
            offset = group_offset

    except error:
        fatal_error(error)


#
# Radix-3
#
fn _radix_3_fft[
    depth: Int, //, twiddle: Bool
](
    radix: Int,
    sofar_radix: Int,
    remain_radix: Int,
    owned group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    mut result: Matrix[DType.float64, depth, complex=True],
):
    try:
        alias theta = 2 * pi / 3
        var c3_1 = cos(theta) - 1
        var c3_2 = sin(theta)

        var offset = group_offset
        var indices = InlineArray[Int, 3](fill=0)

        for _ in range(remain_radix):
            for block in range(radix):
                indices[block] = offset
                offset += sofar_radix

            var value_0 = result[0, indices[0], 0]
            var value_1 = result[0, indices[1], 0]
            var value_2 = result[0, indices[2], 0]

            @parameter
            if twiddle:
                value_1 *= twiddle_factors[1]
                value_2 *= twiddle_factors[2]

            var t1 = value_1 + value_2
            var z0 = value_0 + t1
            var m1 = c3_1 * t1
            var m2 = (c3_2 * (value_1.imaginary() - value_2.imaginary()), c3_2 * (value_2.real() - value_1.real()))
            var s1 = z0 + m1

            result[0, indices[0], 0] = z0
            result[0, indices[1], 0] = s1 + m2
            result[0, indices[2], 0] = s1 - m2

            group_offset = group_offset + sofar_radix * radix
            offset = group_offset

    except error:
        fatal_error(error)


#
# Radix-4
#
fn _radix_4_fft[
    depth: Int, //, twiddle: Bool
](
    radix: Int,
    sofar_radix: Int,
    remain_radix: Int,
    owned group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    mut result: Matrix[DType.float64, depth, complex=True],
):
    try:
        var offset = group_offset
        var indices = InlineArray[Int, 4](fill=0)

        for _ in range(remain_radix):
            for block in range(radix):
                indices[block] = offset
                offset += sofar_radix

            var value_0 = result[0, indices[0], 0]
            var value_1 = result[0, indices[1], 0]
            var value_2 = result[0, indices[2], 0]
            var value_3 = result[0, indices[3], 0]

            @parameter
            if twiddle:
                value_1 *= twiddle_factors[1]
                value_2 *= twiddle_factors[2]
                value_3 *= twiddle_factors[3]

            var t1 = value_0 + value_2
            var m2 = value_0 - value_2
            var t2 = value_1 + value_3
            var m3 = (value_1.imaginary() - value_3.imaginary(), value_3.real() - value_1.real())

            result[0, indices[0], 0] = t1 + t2
            result[0, indices[1], 0] = m2 + m3
            result[0, indices[2], 0] = t1 - t2
            result[0, indices[3], 0] = m2 - m3

            group_offset = group_offset + sofar_radix * radix
            offset = group_offset

    except error:
        fatal_error(error)


#
# Radix-5
#
fn _radix_5_fft[
    depth: Int, //, twiddle: Bool
](
    radix: Int,
    sofar_radix: Int,
    remain_radix: Int,
    owned group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    mut result: Matrix[DType.float64, depth, complex=True],
):
    try:
        alias theta = 2 * pi / 5
        var c5_1 = (cos(theta) + cos(2 * theta)) / 2 - 1
        var c5_2 = (cos(theta) - cos(2 * theta)) / 2
        var c5_3 = -sin(theta)
        var c5_4 = -(sin(theta) + sin(2 * theta))
        var c5_5 = sin(theta) - sin(2 * theta)

        var offset = group_offset
        var indices = InlineArray[Int, 5](fill=0)

        for _ in range(remain_radix):
            for block in range(radix):
                indices[block] = offset
                offset += sofar_radix

            var value_0 = result[0, indices[0], 0]
            var value_1 = result[0, indices[1], 0]
            var value_2 = result[0, indices[2], 0]
            var value_3 = result[0, indices[3], 0]
            var value_4 = result[0, indices[4], 0]

            @parameter
            if twiddle:
                value_1 *= twiddle_factors[1]
                value_2 *= twiddle_factors[2]
                value_3 *= twiddle_factors[3]
                value_4 *= twiddle_factors[4]

            var t1 = value_1 + value_4
            var t2 = value_2 + value_3
            var t3 = value_1 - value_4
            var t4 = value_3 - value_2
            var t5 = t1 + t2

            value_0 += t5

            var m1 = c5_1 * t5
            var m2 = c5_2 * (t1 - t2)
            var m3: ScalarNumber[DType.float64, complex=True] = (-c5_3 * (t3.imaginary() + t4.imaginary()), c5_3 * (t3.real() + t4.real()))
            var m4 = (-c5_4 * t4.imaginary(), c5_4 * t4.real())
            var m5 = (-c5_5 * t3.imaginary(), c5_5 * t3.real())

            var s2 = value_0 + m1 + m2
            var s4 = value_0 + m1 - m2
            var s3 = m3 - m4
            var s5 = m3 + m5

            result[0, indices[0], 0] = value_0
            result[0, indices[1], 0] = s2 + s3
            result[0, indices[2], 0] = s4 + s5
            result[0, indices[3], 0] = s4 - s5
            result[0, indices[4], 0] = s2 - s3

            group_offset = group_offset + sofar_radix * radix
            offset = group_offset

    except error:
        fatal_error(error)


#
# Radix-7
#
fn _radix_7_fft[
    depth: Int, //, twiddle: Bool
](
    radix: Int,
    sofar_radix: Int,
    remain_radix: Int,
    owned group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    mut result: Matrix[DType.float64, depth, complex=True],
):
    try:
        alias theta = 2 * pi / 7
        var c7_1 = 0.222520933956314404288902564496794759466355569
        var c7_2 = 0.900968867902419126236102319507445051165919162
        var c7_3 = 0.623489801858733530525004884004239810632274731  # cos(theta)
        var c7_4 = 0.433883739117558120475768332848358754609990728
        var c7_5 = 0.781831482468029808708444526674057750232334519  # sin(theta)
        var c7_6 = 0.974927912181823607018131682993931217232785801

        var offset = group_offset
        var indices = InlineArray[Int, 7](fill=0)

        for _ in range(remain_radix):
            for block in range(radix):
                indices[block] = offset
                offset += sofar_radix

            var value_0 = result[0, indices[0], 0]
            var value_1 = result[0, indices[1], 0]
            var value_2 = result[0, indices[2], 0]
            var value_3 = result[0, indices[3], 0]
            var value_4 = result[0, indices[4], 0]
            var value_5 = result[0, indices[5], 0]
            var value_6 = result[0, indices[6], 0]

            @parameter
            if twiddle:
                value_1 *= twiddle_factors[1]
                value_2 *= twiddle_factors[2]
                value_3 *= twiddle_factors[3]
                value_4 *= twiddle_factors[4]
                value_5 *= twiddle_factors[5]
                value_6 *= twiddle_factors[6]

            # result[0, indices[0], 0] =
            # result[0, indices[1], 0] =
            # result[0, indices[2], 0] =
            # result[0, indices[3], 0] =
            # result[0, indices[4], 0] =

            group_offset = group_offset + sofar_radix * radix
            offset = group_offset

    except error:
        fatal_error(error)


#
# Radix-8
#
fn _radix_8_fft[
    depth: Int, //, twiddle: Bool
](
    radix: Int,
    sofar_radix: Int,
    remain_radix: Int,
    owned group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    mut result: Matrix[DType.float64, depth, complex=True],
):
    pass


#
# Radix-10
#
fn _radix_10_fft[
    depth: Int, //, twiddle: Bool
](
    radix: Int,
    sofar_radix: Int,
    remain_radix: Int,
    owned group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    mut result: Matrix[DType.float64, depth, complex=True],
):
    pass


#
# Radix-prime
#
fn _radix_prime_fft[
    depth: Int, //, twiddle: Bool
](
    radix: Int,
    sofar_radix: Int,
    remain_radix: Int,
    owned group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    mut result: Matrix[DType.float64, depth, complex=True],
):
    pass
