#
# fft.mojo
# mosaic
#
# Created by Christian Bator on 04/01/2025
#

from math import pi, cos, sin, sqrt
from bit import is_power_of_two, bit_width
from memory import memset_zero
from algorithm import parallelize

from mosaic.utility import fatal_error, print_list


#
# FFT
#
fn fft[dtype: DType, depth: Int, complex: Bool, //](matrix: Matrix[dtype, depth, complex=complex]) -> Matrix[DType.float64, depth, complex=True]:
    """ """
    var result = Matrix[DType.float64, depth, complex=True](rows=matrix.rows(), cols=matrix.cols())

    #
    # Row-wise mixed-radix FFT
    #
    print("\n--- Rows ---")
    var row_plan = _FactorPlan(matrix.cols())
    print("\nActual:  ", end="")
    print_list(row_plan.actual)
    print("Sofar:   ", end="")
    print_list(row_plan.sofar)
    print("Remain:  ", end="")
    print_list(row_plan.remain)
    print()

    _permute(matrix, row_plan, result)

    @parameter
    for component in range(depth):

        @parameter
        fn process_row_wise(row: Int):
            _dispatch[twiddle=False](row_plan.actual[1], row_plan.sofar[1], row_plan.remain[1], row, component, result)

            for i in range(2, len(row_plan.actual)):
                _dispatch[twiddle=True](row_plan.actual[i], row_plan.sofar[i], row_plan.remain[i], row, component, result)

        parallelize[process_row_wise](result.rows())

    #
    # Col-wise mixed-radix FFT
    #
    print("--- Cols ---")
    result.transpose()

    var col_plan = _FactorPlan(result.cols())
    print("\nActual:  ", end="")
    print_list(col_plan.actual)
    print("Sofar:   ", end="")
    print_list(col_plan.sofar)
    print("Remain:  ", end="")
    print_list(col_plan.remain)
    print()

    _permute(result.copy(), col_plan, result)

    @parameter
    for component in range(depth):

        @parameter
        fn process_col_wise(row: Int):
            _dispatch[twiddle=False](col_plan.actual[1], col_plan.sofar[1], col_plan.remain[1], row, component, result)

            for i in range(2, len(col_plan.actual)):
                _dispatch[twiddle=True](col_plan.actual[i], col_plan.sofar[i], col_plan.remain[i], row, component, result)

        parallelize[process_col_wise](result.rows())

    result.transpose()

    return result^


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

        alias radices = InlineArray[Int, 5](2, 3, 4, 5, 7)

        var i = len(radices) - 1

        # Factor by supported radices in decreasing order
        while (N > 1) and (i >= 0):
            if (N % radices[i]) == 0:
                N //= radices[i]
                factors.append(radices[i])
            else:
                i -= 1

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
    dtype: DType, depth: Int, complex: Bool, //
](matrix: Matrix[dtype, depth, complex=complex], plan: _FactorPlan, mut result: Matrix[DType.float64, depth, complex=True]):
    try:
        var N = matrix.cols()
        var indices = List[Int](capacity=N)
        var count = NumericArray[DType.index](count=len(plan.actual))

        var k = 0
        for _ in range(N - 1):
            indices.append(k)

            var j = 1
            k += plan.remain[j]
            count[1] += 1
            while count[j] >= plan.actual[j]:
                count[j] = 0
                k = k - plan.remain[j - 1] + plan.remain[j + 1]
                j += 1
                count[j] += 1

        indices.append(N - 1)

        @parameter
        for component in range(depth):

            @parameter
            fn move_elements(row: Int):
                try:
                    for i in range(len(indices)):
                        result[row, i, component] = matrix[row, indices[i], component].as_complex[DType.float64]()
                except error:
                    fatal_error(error)

            parallelize[move_elements](matrix.rows())

    except error:
        fatal_error(error)


#
# Dispatch
#
fn _dispatch[
    depth: Int, //, *, twiddle: Bool
](radix: Int, sofar_radix: Int, remain_radix: Int, row: Int, component: Int, mut result: Matrix[DType.float64, depth, complex=True]):
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
                _radix_2_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, row, component, result)
            elif radix == 3:
                _radix_3_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, row, component, result)
            elif radix == 4:
                _radix_4_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, row, component, result)
            elif radix == 5:
                _radix_5_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, row, component, result)
            elif radix == 7:
                _radix_7_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, row, component, result)
            else:
                _radix_prime_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, row, component, result)

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
    group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    row: Int,
    component: Int,
    mut result: Matrix[DType.float64, depth, complex=True],
):
    try:
        var group_offset_copy = group_offset
        var offset = group_offset_copy
        var indices = InlineArray[Int, 2](fill=0)

        for _ in range(remain_radix):
            for block in range(radix):
                indices[block] = offset
                offset += sofar_radix

            var value_0 = result[row, indices[0], component]
            var value_1 = result[row, indices[1], component]

            @parameter
            if twiddle:
                value_1 *= twiddle_factors[1]

            result[row, indices[0], component] = value_0 + value_1
            result[row, indices[1], component] = value_0 - value_1

            group_offset_copy = group_offset_copy + sofar_radix * radix
            offset = group_offset_copy

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
    group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    row: Int,
    component: Int,
    mut result: Matrix[DType.float64, depth, complex=True],
):
    try:
        alias theta = 2 * pi / 3
        var c3_1 = cos(theta) - 1
        var c3_2 = sin(theta)

        var group_offset_copy = group_offset
        var offset = group_offset_copy
        var indices = InlineArray[Int, 3](fill=0)

        for _ in range(remain_radix):
            for block in range(radix):
                indices[block] = offset
                offset += sofar_radix

            var value_0 = result[row, indices[0], component]
            var value_1 = result[row, indices[1], component]
            var value_2 = result[row, indices[2], component]

            @parameter
            if twiddle:
                value_1 *= twiddle_factors[1]
                value_2 *= twiddle_factors[2]

            var t1 = value_1 + value_2
            var z0 = value_0 + t1
            var m1 = c3_1 * t1
            var m2 = (c3_2 * (value_1.imaginary() - value_2.imaginary()), c3_2 * (value_2.real() - value_1.real()))
            var s1 = z0 + m1

            result[row, indices[0], component] = z0
            result[row, indices[1], component] = s1 + m2
            result[row, indices[2], component] = s1 - m2

            group_offset_copy = group_offset_copy + sofar_radix * radix
            offset = group_offset_copy

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
    group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    row: Int,
    component: Int,
    mut result: Matrix[DType.float64, depth, complex=True],
):
    try:
        var group_offset_copy = group_offset
        var offset = group_offset_copy
        var indices = InlineArray[Int, 4](fill=0)

        for _ in range(remain_radix):
            for block in range(radix):
                indices[block] = offset
                offset += sofar_radix

            var value_0 = result[row, indices[0], component]
            var value_1 = result[row, indices[1], component]
            var value_2 = result[row, indices[2], component]
            var value_3 = result[row, indices[3], component]

            @parameter
            if twiddle:
                value_1 *= twiddle_factors[1]
                value_2 *= twiddle_factors[2]
                value_3 *= twiddle_factors[3]

            var t1 = value_0 + value_2
            var m2 = value_0 - value_2
            var t2 = value_1 + value_3
            var m3 = (value_1.imaginary() - value_3.imaginary(), value_3.real() - value_1.real())

            result[row, indices[0], component] = t1 + t2
            result[row, indices[1], component] = m2 + m3
            result[row, indices[2], component] = t1 - t2
            result[row, indices[3], component] = m2 - m3

            group_offset_copy = group_offset_copy + sofar_radix * radix
            offset = group_offset_copy

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
    group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    row: Int,
    component: Int,
    mut result: Matrix[DType.float64, depth, complex=True],
):
    try:
        alias theta = 2 * pi / 5
        var c5_1 = (cos(theta) + cos(2 * theta)) / 2 - 1
        var c5_2 = (cos(theta) - cos(2 * theta)) / 2
        var c5_3 = -sin(theta)
        var c5_4 = -(sin(theta) + sin(2 * theta))
        var c5_5 = sin(theta) - sin(2 * theta)

        var group_offset_copy = group_offset
        var offset = group_offset_copy
        var indices = InlineArray[Int, 5](fill=0)

        for _ in range(remain_radix):
            for block in range(radix):
                indices[block] = offset
                offset += sofar_radix

            var value_0 = result[row, indices[0], component]
            var value_1 = result[row, indices[1], component]
            var value_2 = result[row, indices[2], component]
            var value_3 = result[row, indices[3], component]
            var value_4 = result[row, indices[4], component]

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

            result[row, indices[0], component] = value_0
            result[row, indices[1], component] = s2 + s3
            result[row, indices[2], component] = s4 + s5
            result[row, indices[3], component] = s4 - s5
            result[row, indices[4], component] = s2 - s3

            group_offset_copy = group_offset_copy + sofar_radix * radix
            offset = group_offset_copy

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
    group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    row: Int,
    component: Int,
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

        var group_offset_copy = group_offset
        var offset = group_offset_copy
        var indices = InlineArray[Int, 7](fill=0)

        for _ in range(remain_radix):
            for block in range(radix):
                indices[block] = offset
                offset += sofar_radix

            var value_0 = result[row, indices[0], component]
            var value_1 = result[row, indices[1], component]
            var value_2 = result[row, indices[2], component]
            var value_3 = result[row, indices[3], component]
            var value_4 = result[row, indices[4], component]
            var value_5 = result[row, indices[5], component]
            var value_6 = result[row, indices[6], component]

            @parameter
            if twiddle:
                value_1 *= twiddle_factors[1]
                value_2 *= twiddle_factors[2]
                value_3 *= twiddle_factors[3]
                value_4 *= twiddle_factors[4]
                value_5 *= twiddle_factors[5]
                value_6 *= twiddle_factors[6]

            var com_16_add = value_1 + value_6
            var re_16_add = com_16_add.real()
            var im_16_add = com_16_add.imaginary()
            var re_61_sub = value_6.real() - value_1.real()
            var im_16_sub = value_1.imaginary() - value_6.imaginary()

            var com_25_add = value_2 + value_5
            var re_25_add = com_25_add.real()
            var im_25_add = com_25_add.imaginary()
            var re_52_sub = value_5.real() - value_2.real()
            var im_25_sub = value_2.imaginary() - value_5.imaginary()

            var com_34_add = value_3 + value_4
            var re_34_add = com_34_add.real()
            var im_34_add = com_34_add.imaginary()
            var re_43_sub = value_4.real() - value_3.real()
            var im_34_sub = value_3.imaginary() - value_4.imaginary()

            var re_16_result_x = c7_5 * im_16_sub + c7_6 * im_25_sub + c7_4 * im_34_sub
            var re_16_result_y = c7_3 * re_16_add - c7_1 * re_25_add - c7_2 * re_34_add + value_0.real()
            var im_16_result_x = c7_5 * re_61_sub + c7_6 * re_52_sub + c7_4 * re_43_sub
            var im_16_result_y = c7_3 * im_16_add - c7_1 * im_25_add - c7_2 * im_34_add + value_0.imaginary()

            var re_25_result_x = c7_6 * im_16_sub - c7_5 * im_34_sub - c7_4 * im_25_sub
            var re_25_result_y = c7_3 * re_34_add - c7_2 * re_25_add - c7_1 * re_16_add + value_0.real()
            var im_25_result_x = c7_6 * re_61_sub - c7_5 * re_43_sub - c7_4 * re_52_sub
            var im_25_result_y = c7_3 * im_34_add - c7_2 * im_25_add - c7_1 * im_16_add + value_0.imaginary()

            var re_34_result_x = c7_4 * im_16_sub + c7_6 * im_34_sub - c7_5 * im_25_sub
            var re_34_result_y = c7_3 * re_25_add - c7_1 * re_34_add - c7_2 * re_16_add + value_0.real()
            var im_34_result_x = c7_4 * re_61_sub + c7_6 * re_43_sub - c7_5 * re_52_sub
            var im_34_result_y = c7_3 * im_25_add - c7_1 * im_34_add - c7_2 * im_16_add + value_0.imaginary()

            result[row, indices[0], component] = value_0 + com_16_add + com_25_add + com_34_add
            result[row, indices[1], component] = (re_16_result_x + re_16_result_y, im_16_result_x + im_16_result_y)
            result[row, indices[2], component] = (re_25_result_x + re_25_result_y, im_25_result_x + im_25_result_y)
            result[row, indices[3], component] = (re_34_result_x + re_34_result_y, im_34_result_x + im_34_result_y)
            result[row, indices[4], component] = (re_34_result_y - re_34_result_x, im_34_result_y - im_34_result_x)
            result[row, indices[5], component] = (re_25_result_y - re_25_result_x, im_25_result_y - im_25_result_x)
            result[row, indices[6], component] = (re_16_result_y - re_16_result_x, im_16_result_y - im_16_result_x)

            group_offset_copy = group_offset_copy + sofar_radix * radix
            offset = group_offset_copy

    except error:
        fatal_error(error)


#
# Radix-prime
#
fn _radix_prime_fft[
    depth: Int, //, twiddle: Bool
](
    radix: Int,
    sofar_radix: Int,
    remain_radix: Int,
    group_offset: Int,
    twiddle_factors: NumericArray[DType.float64, complex=True],
    row: Int,
    component: Int,
    mut result: Matrix[DType.float64, depth, complex=True],
):
    pass
