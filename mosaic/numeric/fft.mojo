#
# fft.mojo
# mosaic
#
# Created by Christian Bator on 04/01/2025
#

from math import pi, cos, sin, sqrt, ceildiv
from memory import memset_zero
from algorithm import parallelize


#
# FFT
#
fn fft[
    dtype: DType, depth: Int, complex: Bool, //, *, inverse: Bool = False
](matrix: Matrix[dtype, depth, complex=complex]) -> Matrix[DType.float64, depth, complex=True]:
    """ """
    var result = Matrix[DType.float64, depth, complex=True](rows=matrix.rows(), cols=matrix.cols())

    #
    # Row-wise mixed-radix FFT
    #
    if result.cols() > 1:
        var row_plan = _FactorPlan(result.cols())
        _permute[flip_imaginary_sign=inverse](matrix, row_plan, result)

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
    if result.rows() > 1:
        result.transpose()

        var col_plan = _FactorPlan(result.cols())
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

    @parameter
    if inverse:

        @parameter
        fn scale[width: Int](value: Number[DType.float64, width, complex=True]) -> Number[DType.float64, width, complex=True]:
            return Number[DType.float64, width, complex=True](real=value.real() / result.strided_count(), imaginary=-value.imaginary() / result.strided_count())

        result.for_each[scale]()

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
    dtype: DType, depth: Int, complex: Bool, //, *, flip_imaginary_sign: Bool = False
](matrix: Matrix[dtype, depth, complex=complex], plan: _FactorPlan, mut result: Matrix[DType.float64, depth, complex=True]):
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
            for i in range(len(indices)):
                var value = matrix._strided_load(row=row, col=indices[i], component=component).as_complex[DType.float64]()

                @parameter
                if flip_imaginary_sign:
                    result._strided_store(
                        ScalarNumber[DType.float64, complex=True](real=value.real(), imaginary=-value.imaginary()), row=row, col=i, component=component
                    )
                else:
                    result._strided_store(value, row=row, col=i, component=component)

        parallelize[move_elements](matrix.rows())


#
# Dispatch
#
fn _dispatch[
    depth: Int, //, *, twiddle: Bool
](radix: Int, sofar_radix: Int, remain_radix: Int, row: Int, component: Int, mut result: Matrix[DType.float64, depth, complex=True]):
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
    var group_offset_copy = group_offset
    var offset = group_offset_copy
    var indices = InlineArray[Int, 2](fill=0)

    for _ in range(remain_radix):
        for block in range(radix):
            indices[block] = offset
            offset += sofar_radix

        var value_0 = result._strided_load(row=row, col=indices[0], component=component)
        var value_1 = result._strided_load(row=row, col=indices[1], component=component)

        @parameter
        if twiddle:
            value_1 *= twiddle_factors[1]

        result._strided_store(value_0 + value_1, row=row, col=indices[0], component=component)
        result._strided_store(value_0 - value_1, row=row, col=indices[0], component=component)

        group_offset_copy = group_offset_copy + sofar_radix * radix
        offset = group_offset_copy


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

        var value_0 = result._strided_load(row=row, col=indices[0], component=component)
        var value_1 = result._strided_load(row=row, col=indices[1], component=component)
        var value_2 = result._strided_load(row=row, col=indices[2], component=component)

        @parameter
        if twiddle:
            value_1 *= twiddle_factors[1]
            value_2 *= twiddle_factors[2]

        var t1 = value_1 + value_2
        var z0 = value_0 + t1
        var m1 = c3_1 * t1
        var m2 = (c3_2 * (value_1.imaginary() - value_2.imaginary()), c3_2 * (value_2.real() - value_1.real()))
        var s1 = z0 + m1

        result._strided_store(z0, row=row, col=indices[0], component=component)
        result._strided_store(s1 + m2, row=row, col=indices[1], component=component)
        result._strided_store(s1 - m2, row=row, col=indices[2], component=component)

        group_offset_copy = group_offset_copy + sofar_radix * radix
        offset = group_offset_copy


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
    var group_offset_copy = group_offset
    var offset = group_offset_copy
    var indices = InlineArray[Int, 4](fill=0)

    for _ in range(remain_radix):
        for block in range(radix):
            indices[block] = offset
            offset += sofar_radix

        var value_0 = result._strided_load(row=row, col=indices[0], component=component)
        var value_1 = result._strided_load(row=row, col=indices[1], component=component)
        var value_2 = result._strided_load(row=row, col=indices[2], component=component)
        var value_3 = result._strided_load(row=row, col=indices[3], component=component)

        @parameter
        if twiddle:
            value_1 *= twiddle_factors[1]
            value_2 *= twiddle_factors[2]
            value_3 *= twiddle_factors[3]

        var t1 = value_0 + value_2
        var m2 = value_0 - value_2
        var t2 = value_1 + value_3
        var m3 = (value_1.imaginary() - value_3.imaginary(), value_3.real() - value_1.real())

        result._strided_store(t1 + t2, row=row, col=indices[0], component=component)
        result._strided_store(m2 + m3, row=row, col=indices[1], component=component)
        result._strided_store(t1 - t2, row=row, col=indices[2], component=component)
        result._strided_store(m2 - m3, row=row, col=indices[3], component=component)

        group_offset_copy = group_offset_copy + sofar_radix * radix
        offset = group_offset_copy


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

        var value_0 = result._strided_load(row=row, col=indices[0], component=component)
        var value_1 = result._strided_load(row=row, col=indices[1], component=component)
        var value_2 = result._strided_load(row=row, col=indices[2], component=component)
        var value_3 = result._strided_load(row=row, col=indices[3], component=component)
        var value_4 = result._strided_load(row=row, col=indices[4], component=component)

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

        result._strided_store(value_0, row=row, col=indices[0], component=component)
        result._strided_store(s2 + s3, row=row, col=indices[1], component=component)
        result._strided_store(s4 + s5, row=row, col=indices[2], component=component)
        result._strided_store(s4 - s5, row=row, col=indices[3], component=component)
        result._strided_store(s2 - s3, row=row, col=indices[4], component=component)

        group_offset_copy = group_offset_copy + sofar_radix * radix
        offset = group_offset_copy


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

        var value_0 = result._strided_load(row=row, col=indices[0], component=component)
        var value_1 = result._strided_load(row=row, col=indices[1], component=component)
        var value_2 = result._strided_load(row=row, col=indices[2], component=component)
        var value_3 = result._strided_load(row=row, col=indices[3], component=component)
        var value_4 = result._strided_load(row=row, col=indices[4], component=component)
        var value_5 = result._strided_load(row=row, col=indices[5], component=component)
        var value_6 = result._strided_load(row=row, col=indices[6], component=component)

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

        result._strided_store(value_0 + com_16_add + com_25_add + com_34_add, row=row, col=indices[0], component=component)
        result._strided_store((re_16_result_x + re_16_result_y, im_16_result_x + im_16_result_y), row=row, col=indices[1], component=component)
        result._strided_store((re_25_result_x + re_25_result_y, im_25_result_x + im_25_result_y), row=row, col=indices[2], component=component)
        result._strided_store((re_34_result_x + re_34_result_y, im_34_result_x + im_34_result_y), row=row, col=indices[3], component=component)
        result._strided_store((re_34_result_y - re_34_result_x, im_34_result_y - im_34_result_x), row=row, col=indices[4], component=component)
        result._strided_store((re_25_result_y - re_25_result_x, im_25_result_y - im_25_result_x), row=row, col=indices[5], component=component)
        result._strided_store((re_16_result_y - re_16_result_x, im_16_result_y - im_16_result_x), row=row, col=indices[6], component=component)

        group_offset_copy = group_offset_copy + sofar_radix * radix
        offset = group_offset_copy


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
    var group_offset_copy = group_offset
    var offset = group_offset_copy
    var indices = NumericArray[DType.index](count=radix)

    # TODO: Make this better, prepare factors in advance?
    var values = NumericArray[DType.float64, complex=True](count=radix)
    var v_values = NumericArray[DType.float64, complex=True](count=ceildiv(radix, 2))
    var w_values = NumericArray[DType.float64, complex=True](count=ceildiv(radix, 2))
    var trig = NumericArray[DType.float64, complex=True](count=radix)
    var theta = 2 * pi / radix
    var factor = (cos(theta), -sin(theta))

    trig[0] = (1.0, 0.0)
    trig[1] = factor

    for i in range(2, radix):
        trig[i] = factor * trig[i - 1]
    ##

    for _ in range(remain_radix):
        for block in range(radix):
            indices[block] = offset
            offset += sofar_radix

        values[0] = result._strided_load(row=row, col=Int(indices[0]), component=component)

        for block in range(1, radix):

            @parameter
            if twiddle:
                values[block] = twiddle_factors[block] * result._strided_load(row=row, col=Int(indices[block]), component=component)
            else:
                values[block] = result._strided_load(row=row, col=Int(indices[block]), component=component)

        var value_0 = values[0]

        var n = radix
        var half_n = ceildiv(n, 2)
        for j in range(1, half_n):
            v_values[j] = (values[j].real() + values[n - j].real(), values[j].imaginary() - values[n - j].imaginary())
            w_values[j] = (values[j].real() - values[n - j].real(), values[j].imaginary() + values[n - j].imaginary())

        for j in range(1, half_n):
            values[j] = value_0
            values[n - j] = value_0

            var k = j
            for i in range(1, half_n):
                var rere = trig[k].real() * v_values[i].real()
                var imim = trig[k].imaginary() * v_values[i].imaginary()
                var reim = trig[k].real() * w_values[i].imaginary()
                var imre = trig[k].imaginary() * w_values[i].real()

                values[j] += (rere - imim, reim + imre)
                values[n - j] += (rere + imim, reim - imre)

                k += j
                if k >= n:
                    k -= n

        for j in range(1, half_n):
            values[0] = (values[0].real() + v_values[j].real(), values[0].imaginary() + w_values[j].imaginary())

        for j in range(radix):
            result._strided_store(values[j], row=row, col=Int(indices[j]), component=component)

        group_offset_copy = group_offset_copy + sofar_radix * radix
        offset = group_offset_copy
