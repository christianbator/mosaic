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
    var plan = FactorPlan(matrix.cols())
    print("\nActual:  ", end="")
    print_list(plan.actual)
    print("Sofar:   ", end="")
    print_list(plan.sofar)
    print("Remain:  ", end="")
    print_list(plan.remain)
    print()

    _permute(matrix, plan, result)
    print("Reordered: ", end="")
    print(result)
    print()

    _dispatch[twiddle=False](plan.actual[1], plan.sofar[1], plan.remain[1], result)

    for i in range(2, len(plan.actual)):
        _dispatch[twiddle=True](plan.actual[i], plan.sofar[i], plan.remain[i], result)


#
# FactorPlan
#
struct FactorPlan:
    alias radices = InlineArray[Int, 8](1, 2, 3, 4, 5, 7, 8, 10)
    alias max_radix = Self.radices[len(Self.radices) - 1]
    alias max_factor_count = 16

    var factor_count: Int
    var actual: List[Int]
    var sofar: List[Int]
    var remain: List[Int]

    fn __init__(out self, N: Int):
        var factors = _factor(N)
        self.factor_count = len(factors)
        self.actual = List[Int](0)
        self.actual.extend(factors)
        self.sofar = List[Int](0, 1)
        self.remain = List[Int](N, N // self.actual[1])

        for i in range(2, len(self.actual)):
            self.sofar.append(self.sofar[i - 1] * self.actual[i - 1])
            self.remain.append(self.remain[i - 1] // self.actual[i])


fn _factor(owned N: Int) -> List[Int]:
    """
    int i, j, k;
    int nRadix;
    int radices[16];
    int factors[maxFactorCount];

    nRadix = 7;
    radices[1] = 2;
    radices[2] = 3;
    radices[3] = 4;
    radices[4] = 5;
    radices[5] = 7;
    radices[6] = 8;
    radices[7] = 10;

    if (n == 1) {
        j = 1;
        factors[1] = 1;
    }
    else
        j = 0;
    i = nRadix;
    while ((n > 1) && (i > 0)) {
        if ((n % radices[i]) == 0) {
            n = n / radices[i];
            j = j + 1;
            factors[j] = radices[i];
        }
        else
            i = i - 1;
    }
    if (factors[j] == 2) /*substitute factors 2*8 with 4*4 */
    {
        i = j - 1;
        while ((i > 0) && (factors[i] != 8))
            i--;
        if (i > 0) {
            factors[j] = 4;
            factors[i] = 4;
        }
    }
    if (n > 1) {
        for (k = 2; k < sqrt(n) + 1; k++)
            while ((n % k) == 0) {
                n = n / k;
                j = j + 1;
                factors[j] = k;
            }
        if (n > 1) {
            j = j + 1;
            factors[j] = n;
        }
    }

    for (i = 1; i <= j; i++) {
        fact[i] = factors[i];
    }
    *nFact = j;
    """

    var factors = InlineArray[Int, FactorPlan.max_factor_count](fill=0)

    var j = 0

    if N == 1:
        j = 1
        factors[1] = 1

    var i = len(FactorPlan.radices) - 1

    while (N > 1) and (i > 0):
        if (N % FactorPlan.radices[i]) == 0:
            N //= FactorPlan.radices[i]
            j += 1
            factors[j] = FactorPlan.radices[i]
        else:
            i -= 1

    # Substitute factors 2 * 8 with 4 * 4
    if factors[j] == 2:
        i = j - 1
        while (i > 0) and (factors[i] != 8):
            i -= 1

        if i > 0:
            factors[j] = 4
            factors[i] = 4

    if N > 1:
        for k in range(2, sqrt(N) + 1):
            while (N % k) == 0:
                N //= k
                j += 1
                factors[j] = k

        if N > 1:
            j += 1
            factors[j] = N

    var result = List[Int](capacity=j)

    for index in range(1, j + 1):
        result.append(factors[index])

    return result


#
# Permute
#
fn _permute[
    dtype: DType, depth: Int, //
](matrix: Matrix[dtype, depth, complex=False], plan: FactorPlan, mut result: Matrix[DType.float64, depth, complex=True]):
    """
    int count[maxFactorCount];

    for (int i = 1; i <= nFact; i++) {
        count[i] = 0;
    }

    int k = 0;
    for (int i = 0; i <= nPoint - 2; i++) {
        yRe[i] = xRe[k];
        yIm[i] = xIm[k];

        int j = 1;
        k = k + remain[j];
        count[1] = count[1] + 1;

        while (count[j] >= fact[j]) {
            count[j] = 0;
            k = k - remain[j - 1] + remain[j + 1];
            j = j + 1;
            count[j] = count[j] + 1;
        }
    }

    yRe[nPoint - 1] = xRe[nPoint - 1];
    yIm[nPoint - 1] = xIm[nPoint - 1];
    """
    try:
        var N = matrix.cols()

        var count = List[Int](capacity=len(plan.actual))
        memset_zero(count.unsafe_ptr(), len(plan.actual))

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
    var twiddle_factors = InlineArray[ScalarNumber[DType.float64, complex=True], FactorPlan.max_radix](fill=0)
    var omega = 2.0 * pi / (sofar_radix * radix)
    var base_twiddle_factor: ScalarNumber[DType.float64, complex=True] = (cos(omega), -sin(omega))
    var twiddle_factor: ScalarNumber[DType.float64, complex=True] = (1.0, 0.0)

    # var omega = 2.0 * pi / (sofar_radix * radix)
    # var cosw = cos(omega)
    # var sinw = -sin(omega)
    # var tw_re = 1.0
    # var tw_im = 0.0
    # var twiddle_factors_real = InlineArray[Float64, FactorPlan.max_factor_count](fill=0)
    # var twiddle_factors_imaginary = InlineArray[Float64, FactorPlan.max_factor_count](fill=0)

    for group_offset in range(sofar_radix):

        @parameter
        if twiddle:
            twiddle_factors[0] = (1.0, 0.0)
            twiddle_factors[1] = twiddle_factor
            # twiddle_factors_real[0] = 1.0
            # twiddle_factors_imaginary[0] = 0.0
            # twiddle_factors_real[1] = tw_re
            # twiddle_factors_imaginary[1] = tw_im

            for i in range(2, radix):
                twiddle_factors[i] = twiddle_factor * twiddle_factors[i - 1]
                # twiddle_factors[i] = (
                #     tw_re * twiddle_factors[i - 1].real() - tw_im * twiddle_factors[i - 1].imaginary(),
                #     tw_im * twiddle_factors[i - 1].real() + tw_re * twiddle_factors[i - 1].imaginary()
                # )

                # twiddle_factors_real[i] = tw_re * twiddle_factors_real[i - 1] - tw_im * twiddle_factors_imaginary[i - 1]
                # twiddle_factors_imaginary[i] = tw_im * twiddle_factors_real[i - 1] + tw_re * twiddle_factors_imaginary[i - 1]

            twiddle_factor *= base_twiddle_factor
            # var temp = cosw * tw_re - sinw * tw_im
            # tw_im = sinw * tw_re + cosw * tw_im
            # tw_re = temp

        if radix == 1:
            pass
        if radix == 2:
            _radix_2_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, result)
        elif radix == 3:
            _radix_3_fft[twiddle](radix, sofar_radix, remain_radix, group_offset, twiddle_factors, result)
        elif radix == 4:
            pass
        elif radix == 5:
            pass
        elif radix == 7:
            pass
        elif radix == 8:
            pass
        elif radix == 10:
            pass
        else:
            pass


fn _radix_2_fft[
    depth: Int, //, twiddle: Bool
](
    radix: Int,
    sofar_radix: Int,
    remain_radix: Int,
    owned group_offset: Int,
    twiddle_factors: InlineArray[ScalarNumber[DType.float64, complex=True], FactorPlan.max_radix],
    mut result: Matrix[DType.float64, depth, complex=True],
):
    """
    double ar0 = yRe[yIndx[0]];
    double ai0 = yIm[yIndx[0]];

    double yr = yRe[yIndx[1]];
    double yi = yIm[yIndx[1]];

    double twr = twiddleRe[1];
    double twi = twiddleIm[1];
    double ar1 = twr * yr - twi * yi;
    double ai1 = twr * yi + twi * yr;

    yRe[yIndx[0]] = ar0 + ar1;
    yRe[yIndx[1]] = ar0 - ar1;
    yIm[yIndx[0]] = ai0 + ai1;
    yIm[yIndx[1]] = ai0 - ai1;
    """
    print("FFT 2")
    try:
        var offset = group_offset
        for _ in range(remain_radix):
            var indices = InlineArray[Int, 2](fill=0)

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


fn _radix_3_fft[
    depth: Int, //, twiddle: Bool
](
    radix: Int,
    sofar_radix: Int,
    remain_radix: Int,
    owned group_offset: Int,
    twiddle_factors: InlineArray[ScalarNumber[DType.float64, complex=True], FactorPlan.max_radix],
    mut result: Matrix[DType.float64, depth, complex=True],
):
    """
    ar0 = yRe[yIndx[0]];
    ai0 = yIm[yIndx[0]];

    yr = yRe[yIndx[1]];
    yi = yIm[yIndx[1]];
    twr = twiddleRe[1];
    twi = twiddleIm[1];
    ar1 = twr * yr - twi * yi;
    ai1 = twr * yi + twi * yr;

    yr = yRe[yIndx[2]];
    yi = yIm[yIndx[2]];
    twr = twiddleRe[2];
    twi = twiddleIm[2];
    ar2 = twr * yr - twi * yi;
    ai2 = twr * yi + twi * yr;

    t1_re = ar1 + ar2;
    t1_im = ai1 + ai2;
    zr0 = ar0 + t1_re;
    zi0 = ai0 + t1_im;
    m1_re = c3_1 * t1_re;
    m1_im = c3_1 * t1_im;
    m2_re = c3_2 * (ai1 - ai2);
    m2_im = c3_2 * (ar2 - ar1);

    s1_re = zr0 + m1_re;
    s1_im = zi0 + m1_im;

    yRe[yIndx[0]] = zr0;
    yIm[yIndx[0]] = zi0;
    yRe[yIndx[1]] = s1_re + m2_re;
    yIm[yIndx[1]] = s1_im + m2_im;
    yRe[yIndx[2]] = s1_re - m2_re;
    yIm[yIndx[2]] = s1_im - m2_im;
    """
    print("FFT 3")

    @parameter
    if twiddle:
        print("Twiddles: ", end="")
        for i in range(len(twiddle_factors)):
            print(twiddle_factors[i], end=", ")

        print()

    alias c3_1 = -1.5  # cos(2 * pi / 3) - 1
    alias c3_2 = 0.8660254037844386467637231707  # sin(2 * pi / 3)

    try:
        var offset = group_offset
        for _ in range(remain_radix):
            var indices = InlineArray[Int, 3](fill=0)

            for block in range(radix):
                indices[block] = offset
                offset += sofar_radix

            var value_0 = result[0, indices[0], 0]
            var value_1 = result[0, indices[1], 0]
            var value_2 = result[0, indices[2], 0]

            # print("a0:", value_0)
            # print("a1:", value_1)
            # print("a2:", value_2)

            @parameter
            if twiddle:
                value_1 *= twiddle_factors[1]
                value_2 *= twiddle_factors[2]

            var t1 = value_1 + value_2
            # print("t1:", t1)
            var z0 = value_0 + t1
            # print("z0:", z0)
            var m1 = c3_1 * t1
            # print("m1:", m1)
            var m2: ScalarNumber[DType.float64, complex=True] = (c3_2 * (value_1.imaginary() - value_2.imaginary()), c3_2 * (value_2.real() - value_1.real()))
            # print("m2:", m2)
            var s1 = z0 + m1
            # print("s1:", s1)

            result[0, indices[0], 0] = z0
            result[0, indices[1], 0] = s1 + m2
            result[0, indices[2], 0] = s1 - m2

            group_offset = group_offset + sofar_radix * radix
            offset = group_offset
    except error:
        fatal_error(error)


# fn optimal_fft_size(N: Int) -> Int:
#     for i in range(len(_optimal_fft_sizes)):
#         if N <= _optimal_fft_sizes[i]:
#             return _optimal_fft_sizes[i]

#     return N

# Multiples of (1, 2, 3, 5)
# alias _optimal_fft_sizes = List[Int](
#     1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
#     50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160,
#     162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375,
#     384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720,
#     729, 750, 768, 800, 810, 864, 900, 960, 972, 1000, 1024, 1080, 1125, 1152, 1200,
#     1215, 1250, 1280, 1296, 1350, 1440, 1458, 1500, 1536, 1600, 1620, 1728, 1800, 1875,
#     1920, 1944, 2000, 2025, 2048, 2160, 2187, 2250, 2304, 2400, 2430, 2500, 2560, 2592,
#     2700, 2880, 2916, 3000, 3072, 3125, 3200, 3240, 3375, 3456, 3600, 3645, 3750, 3840,
#     3888, 4000, 4050, 4096, 4320, 4374, 4500, 4608, 4800, 4860, 5000, 5120, 5184, 5400,
#     5625, 5760, 5832, 6000, 6075, 6144, 6250, 6400, 6480, 6561, 6750, 6912, 7200, 7290,
#     7500, 7680, 7776, 8000, 8100, 8192, 8640, 8748, 9000, 9216, 9375, 9600, 9720, 10000,
#     10125, 10240, 10368, 10800, 10935, 11250, 11520, 11664, 12000, 12150, 12288, 12500,
#     12800, 12960, 13122, 13500, 13824, 14400, 14580, 15000, 15360, 15552, 15625, 16000,
#     16200, 16384, 16875, 17280, 17496, 18000, 18225, 18432, 18750, 19200, 19440, 19683,
#     20000, 20250, 20480, 20736, 21600, 21870, 22500, 23040, 23328, 24000, 24300, 24576,
#     25000, 25600, 25920, 26244, 27000, 27648, 28125, 28800, 29160, 30000, 30375, 30720,
#     31104, 31250, 32000, 32400, 32768, 32805, 33750, 34560, 34992, 36000, 36450, 36864,
#     37500, 38400, 38880, 39366, 40000, 40500, 40960, 41472, 43200, 43740, 45000, 46080,
#     46656, 46875, 48000, 48600, 49152, 50000, 50625, 51200, 51840, 52488, 54000, 54675,
#     55296, 56250, 57600, 58320, 59049, 60000, 60750, 61440, 62208, 62500, 64000, 64800,
#     65536, 65610, 67500, 69120, 69984, 72000, 72900, 73728, 75000, 76800, 77760, 78125,
#     78732, 80000, 81000, 81920, 82944, 84375, 86400, 87480, 90000, 91125, 92160, 93312,
#     93750, 96000, 97200, 98304, 98415, 100000, 101250, 102400, 103680, 104976, 108000,
#     109350, 110592, 112500, 115200, 116640, 118098, 120000, 121500, 122880, 124416, 125000,
#     128000, 129600, 131072, 131220, 135000, 138240, 139968, 140625, 144000, 145800, 147456,
#     150000, 151875, 153600, 155520, 156250, 157464, 160000, 162000, 163840, 164025, 165888,
#     168750, 172800, 174960, 177147, 180000, 182250, 184320, 186624, 187500, 192000, 194400,
#     196608, 196830, 200000, 202500, 204800, 207360, 209952, 216000, 218700, 221184, 225000,
#     230400, 233280, 234375, 236196, 240000, 243000, 245760, 248832, 250000, 253125, 256000,
#     259200, 262144, 262440, 270000, 273375, 276480, 279936, 281250, 288000, 291600, 294912,
#     295245, 300000, 303750, 307200, 311040, 312500, 314928, 320000, 324000, 327680, 328050,
#     331776, 337500, 345600, 349920, 354294, 360000, 364500, 368640, 373248, 375000, 384000,
#     388800, 390625, 393216, 393660, 400000, 405000, 409600, 414720, 419904, 421875, 432000,
#     437400, 442368, 450000, 455625, 460800, 466560, 468750, 472392, 480000, 486000, 491520,
#     492075, 497664, 500000, 506250, 512000, 518400, 524288, 524880, 531441, 540000, 546750,
#     552960, 559872, 562500, 576000, 583200, 589824, 590490, 600000, 607500, 614400, 622080,
#     625000, 629856, 640000, 648000, 655360, 656100, 663552, 675000, 691200, 699840, 703125,
#     708588, 720000, 729000, 737280, 746496, 750000, 759375, 768000, 777600, 781250, 786432,
#     787320, 800000, 810000, 819200, 820125, 829440, 839808, 843750, 864000, 874800, 884736,
#     885735, 900000, 911250, 921600, 933120, 937500, 944784, 960000, 972000, 983040, 984150,
#     995328, 1000000, 1012500, 1024000, 1036800, 1048576, 1049760, 1062882, 1080000, 1093500,
#     1105920, 1119744, 1125000, 1152000, 1166400, 1171875, 1179648, 1180980, 1200000,
#     1215000, 1228800, 1244160, 1250000, 1259712, 1265625, 1280000, 1296000, 1310720,
#     1312200, 1327104, 1350000, 1366875, 1382400, 1399680, 1406250, 1417176, 1440000,
#     1458000, 1474560, 1476225, 1492992, 1500000, 1518750, 1536000, 1555200, 1562500,
#     1572864, 1574640, 1594323, 1600000, 1620000, 1638400, 1640250, 1658880, 1679616,
#     1687500, 1728000, 1749600, 1769472, 1771470, 1800000, 1822500, 1843200, 1866240,
#     1875000, 1889568, 1920000, 1944000, 1953125, 1966080, 1968300, 1990656, 2000000,
#     2025000, 2048000, 2073600, 2097152, 2099520, 2109375, 2125764, 2160000, 2187000,
#     2211840, 2239488, 2250000, 2278125, 2304000, 2332800, 2343750, 2359296, 2361960,
#     2400000, 2430000, 2457600, 2460375, 2488320, 2500000, 2519424, 2531250, 2560000,
#     2592000, 2621440, 2624400, 2654208, 2657205, 2700000, 2733750, 2764800, 2799360,
#     2812500, 2834352, 2880000, 2916000, 2949120, 2952450, 2985984, 3000000, 3037500,
#     3072000, 3110400, 3125000, 3145728, 3149280, 3188646, 3200000, 3240000, 3276800,
#     3280500, 3317760, 3359232, 3375000, 3456000, 3499200, 3515625, 3538944, 3542940,
#     3600000, 3645000, 3686400, 3732480, 3750000, 3779136, 3796875, 3840000, 3888000,
#     3906250, 3932160, 3936600, 3981312, 4000000, 4050000, 4096000, 4100625, 4147200,
#     4194304, 4199040, 4218750, 4251528, 4320000, 4374000, 4423680, 4428675, 4478976,
#     4500000, 4556250, 4608000, 4665600, 4687500, 4718592, 4723920, 4782969, 4800000,
#     4860000, 4915200, 4920750, 4976640, 5000000, 5038848, 5062500, 5120000, 5184000,
#     5242880, 5248800, 5308416, 5314410, 5400000, 5467500, 5529600, 5598720, 5625000,
#     5668704, 5760000, 5832000, 5859375, 5898240, 5904900, 5971968, 6000000, 6075000,
#     6144000, 6220800, 6250000, 6291456, 6298560, 6328125, 6377292, 6400000, 6480000,
#     6553600, 6561000, 6635520, 6718464, 6750000, 6834375, 6912000, 6998400, 7031250,
#     7077888, 7085880, 7200000, 7290000, 7372800, 7381125, 7464960, 7500000, 7558272,
#     7593750, 7680000, 7776000, 7812500, 7864320, 7873200, 7962624, 7971615, 8000000,
#     8100000, 8192000, 8201250, 8294400, 8388608, 8398080, 8437500, 8503056, 8640000,
#     8748000, 8847360, 8857350, 8957952, 9000000, 9112500, 9216000, 9331200, 9375000,
#     9437184, 9447840, 9565938, 9600000, 9720000, 9765625, 9830400, 9841500, 9953280,
#     10000000, 10077696, 10125000, 10240000, 10368000, 10485760, 10497600, 10546875, 10616832,
#     10628820, 10800000, 10935000, 11059200, 11197440, 11250000, 11337408, 11390625, 11520000,
#     11664000, 11718750, 11796480, 11809800, 11943936, 12000000, 12150000, 12288000, 12301875,
#     12441600, 12500000, 12582912, 12597120, 12656250, 12754584, 12800000, 12960000, 13107200,
#     13122000, 13271040, 13286025, 13436928, 13500000, 13668750, 13824000, 13996800, 14062500,
#     14155776, 14171760, 14400000, 14580000, 14745600, 14762250, 14929920, 15000000, 15116544,
#     15187500, 15360000, 15552000, 15625000, 15728640, 15746400, 15925248, 15943230, 16000000,
#     16200000, 16384000, 16402500, 16588800, 16777216, 16796160, 16875000, 17006112, 17280000,
#     17496000, 17578125, 17694720, 17714700, 17915904, 18000000, 18225000, 18432000, 18662400,
#     18750000, 18874368, 18895680, 18984375, 19131876, 19200000, 19440000, 19531250, 19660800,
#     19683000, 19906560, 20000000, 20155392, 20250000, 20480000, 20503125, 20736000, 20971520,
#     20995200, 21093750, 21233664, 21257640, 21600000, 21870000, 22118400, 22143375, 22394880,
#     22500000, 22674816, 22781250, 23040000, 23328000, 23437500, 23592960, 23619600, 23887872,
#     23914845, 24000000, 24300000, 24576000, 24603750, 24883200, 25000000, 25165824, 25194240,
#     25312500, 25509168, 25600000, 25920000, 26214400, 26244000, 26542080, 26572050, 26873856,
#     27000000, 27337500, 27648000, 27993600, 28125000, 28311552, 28343520, 28800000, 29160000,
#     29296875, 29491200, 29524500, 29859840, 30000000, 30233088, 30375000, 30720000, 31104000,
#     31250000, 31457280, 31492800, 31640625, 31850496, 31886460, 32000000, 32400000, 32768000,
#     32805000, 33177600, 33554432, 33592320, 33750000, 34012224, 34171875, 34560000, 34992000,
#     35156250, 35389440, 35429400, 35831808, 36000000, 36450000, 36864000, 36905625, 37324800,
#     37500000, 37748736, 37791360, 37968750, 38263752, 38400000, 38880000, 39062500, 39321600,
#     39366000, 39813120, 39858075, 40000000, 40310784, 40500000, 40960000, 41006250, 41472000,
#     41943040, 41990400, 42187500, 42467328, 42515280, 43200000, 43740000, 44236800, 44286750,
#     44789760, 45000000, 45349632, 45562500, 46080000, 46656000, 46875000, 47185920, 47239200,
#     47775744, 47829690, 48000000, 48600000, 48828125, 49152000, 49207500, 49766400, 50000000,
#     50331648, 50388480, 50625000, 51018336, 51200000, 51840000, 52428800, 52488000, 52734375,
#     53084160, 53144100, 53747712, 54000000, 54675000, 55296000, 55987200, 56250000, 56623104,
#     56687040, 56953125, 57600000, 58320000, 58593750, 58982400, 59049000, 59719680, 60000000,
#     60466176, 60750000, 61440000, 61509375, 62208000, 62500000, 62914560, 62985600, 63281250,
#     63700992, 63772920, 64000000, 64800000, 65536000, 65610000, 66355200, 66430125, 67108864,
#     67184640, 67500000, 68024448, 68343750, 69120000, 69984000, 70312500, 70778880, 70858800,
#     71663616, 72000000, 72900000, 73728000, 73811250, 74649600, 75000000, 75497472, 75582720,
#     75937500, 76527504, 76800000, 77760000, 78125000, 78643200, 78732000, 79626240, 79716150,
#     80000000, 80621568, 81000000, 81920000, 82012500, 82944000, 83886080, 83980800, 84375000,
#     84934656, 85030560, 86400000, 87480000, 87890625, 88473600, 88573500, 89579520, 90000000,
#     90699264, 91125000, 92160000, 93312000, 93750000, 94371840, 94478400, 94921875, 95551488,
#     95659380, 96000000, 97200000, 97656250, 98304000, 98415000, 99532800, 100000000,
#     100663296, 100776960, 101250000, 102036672, 102400000, 102515625, 103680000, 104857600,
#     104976000, 105468750, 106168320, 106288200, 107495424, 108000000, 109350000, 110592000,
#     110716875, 111974400, 112500000, 113246208, 113374080, 113906250, 115200000, 116640000,
#     117187500, 117964800, 118098000, 119439360, 119574225, 120000000, 120932352, 121500000,
#     122880000, 123018750, 124416000, 125000000, 125829120, 125971200, 126562500, 127401984,
#     127545840, 128000000, 129600000, 131072000, 131220000, 132710400, 132860250, 134217728,
#     134369280, 135000000, 136048896, 136687500, 138240000, 139968000, 140625000, 141557760,
#     141717600, 143327232, 144000000, 145800000, 146484375, 147456000, 147622500, 149299200,
#     150000000, 150994944, 151165440, 151875000, 153055008, 153600000, 155520000, 156250000,
#     157286400, 157464000, 158203125, 159252480, 159432300, 160000000, 161243136, 162000000,
#     163840000, 164025000, 165888000, 167772160, 167961600, 168750000, 169869312, 170061120,
#     170859375, 172800000, 174960000, 175781250, 176947200, 177147000, 179159040, 180000000,
#     181398528, 182250000, 184320000, 184528125, 186624000, 187500000, 188743680, 188956800,
#     189843750, 191102976, 191318760, 192000000, 194400000, 195312500, 196608000, 196830000,
#     199065600, 199290375, 200000000, 201326592, 201553920, 202500000, 204073344, 204800000,
#     205031250, 207360000, 209715200, 209952000, 210937500, 212336640, 212576400, 214990848,
#     216000000, 218700000, 221184000, 221433750, 223948800, 225000000, 226492416, 226748160,
#     227812500, 230400000, 233280000, 234375000, 235929600, 236196000, 238878720, 239148450,
#     240000000, 241864704, 243000000, 244140625, 245760000, 246037500, 248832000, 250000000,
#     251658240, 251942400, 253125000, 254803968, 255091680, 256000000, 259200000, 262144000,
#     262440000, 263671875, 265420800, 265720500, 268435456, 268738560, 270000000, 272097792,
#     273375000, 276480000, 279936000, 281250000, 283115520, 283435200, 284765625, 286654464,
#     288000000, 291600000, 292968750, 294912000, 295245000, 298598400, 300000000, 301989888,
#     302330880, 303750000, 306110016, 307200000, 307546875, 311040000, 312500000, 314572800,
#     314928000, 316406250, 318504960, 318864600, 320000000, 322486272, 324000000, 327680000,
#     328050000, 331776000, 332150625, 335544320, 335923200, 337500000, 339738624, 340122240,
#     341718750, 345600000, 349920000, 351562500, 353894400, 354294000, 358318080, 360000000,
#     362797056, 364500000, 368640000, 369056250, 373248000, 375000000, 377487360, 377913600,
#     379687500, 382205952, 382637520, 384000000, 388800000, 390625000, 393216000, 393660000,
#     398131200, 398580750, 400000000, 402653184, 403107840, 405000000, 408146688, 409600000,
#     410062500, 414720000, 419430400, 419904000, 421875000, 424673280, 425152800, 429981696,
#     432000000, 437400000, 439453125, 442368000, 442867500, 447897600, 450000000, 452984832,
#     453496320, 455625000, 460800000, 466560000, 468750000, 471859200, 472392000, 474609375,
#     477757440, 478296900, 480000000, 483729408, 486000000, 488281250, 491520000, 492075000,
#     497664000, 500000000, 503316480, 503884800, 506250000, 509607936, 510183360, 512000000,
#     512578125, 518400000, 524288000, 524880000, 527343750, 530841600, 531441000, 536870912,
#     537477120, 540000000, 544195584, 546750000, 552960000, 553584375, 559872000, 562500000,
#     566231040, 566870400, 569531250, 573308928, 576000000, 583200000, 585937500, 589824000,
#     590490000, 597196800, 597871125, 600000000, 603979776, 604661760, 607500000, 612220032,
#     614400000, 615093750, 622080000, 625000000, 629145600, 629856000, 632812500, 637009920,
#     637729200, 640000000, 644972544, 648000000, 655360000, 656100000, 663552000, 664301250,
#     671088640, 671846400, 675000000, 679477248, 680244480, 683437500, 691200000, 699840000,
#     703125000, 707788800, 708588000, 716636160, 720000000, 725594112, 729000000, 732421875,
#     737280000, 738112500, 746496000, 750000000, 754974720, 755827200, 759375000, 764411904,
#     765275040, 768000000, 777600000, 781250000, 786432000, 787320000, 791015625, 796262400,
#     797161500, 800000000, 805306368, 806215680, 810000000, 816293376, 819200000, 820125000,
#     829440000, 838860800, 839808000, 843750000, 849346560, 850305600, 854296875, 859963392,
#     864000000, 874800000, 878906250, 884736000, 885735000, 895795200, 900000000, 905969664,
#     906992640, 911250000, 921600000, 922640625, 933120000, 937500000, 943718400, 944784000,
#     949218750, 955514880, 956593800, 960000000, 967458816, 972000000, 976562500, 983040000,
#     984150000, 995328000, 996451875, 1000000000, 1006632960, 1007769600, 1012500000,
#     1019215872, 1020366720, 1024000000, 1025156250, 1036800000, 1048576000, 1049760000,
#     1054687500, 1061683200, 1062882000, 1073741824, 1074954240, 1080000000, 1088391168,
#     1093500000, 1105920000, 1107168750, 1119744000, 1125000000, 1132462080, 1133740800,
#     1139062500, 1146617856, 1152000000, 1166400000, 1171875000, 1179648000, 1180980000,
#     1194393600, 1195742250, 1200000000, 1207959552, 1209323520, 1215000000, 1220703125,
#     1224440064, 1228800000, 1230187500, 1244160000, 1250000000, 1258291200, 1259712000,
#     1265625000, 1274019840, 1275458400, 1280000000, 1289945088, 1296000000, 1310720000,
#     1312200000, 1318359375, 1327104000, 1328602500, 1342177280, 1343692800, 1350000000,
#     1358954496, 1360488960, 1366875000, 1382400000, 1399680000, 1406250000, 1415577600,
#     1417176000, 1423828125, 1433272320, 1440000000, 1451188224, 1458000000, 1464843750,
#     1474560000, 1476225000, 1492992000, 1500000000, 1509949440, 1511654400, 1518750000,
#     1528823808, 1530550080, 1536000000, 1537734375, 1555200000, 1562500000, 1572864000,
#     1574640000, 1582031250, 1592524800, 1594323000, 1600000000, 1610612736, 1612431360,
#     1620000000, 1632586752, 1638400000, 1640250000, 1658880000, 1660753125, 1677721600,
#     1679616000, 1687500000, 1698693120, 1700611200, 1708593750, 1719926784, 1728000000,
#     1749600000, 1757812500, 1769472000, 1771470000, 1791590400, 1800000000, 1811939328,
#     1813985280, 1822500000, 1843200000, 1845281250, 1866240000, 1875000000, 1887436800,
#     1889568000, 1898437500, 1911029760, 1913187600, 1920000000, 1934917632, 1944000000,
#     1953125000, 1966080000, 1968300000, 1990656000, 1992903750, 2000000000, 2013265920,
#     2015539200, 2025000000, 2038431744, 2040733440, 2048000000, 2050312500, 2073600000,
#     2097152000, 2099520000, 2109375000, 2123366400, 2125764000
# )

# fn _factor(N: Int) -> List[Int]:
#     var N_factoring = N
#     var factors = List[Int]()

#     # Small number base case
#     if N_factoring <= 5:
#         factors.append(N_factoring)
#         return factors

#     # Heuristic starting factor
#     var factor = (((N_factoring - 1) ^ N_factoring) + 1) >> 1

#     if factor > 1:
#         factors.append(factor)
#         N_factoring //= factor

#     # Now try factors from 3 upwards incrementing by 2 (odd numbers)
#     factor = 3
#     while N_factoring > 1:
#         if N_factoring % factor == 0:
#             factors.append(factor)
#             N_factoring //= factor
#         else:
#             factor += 2

#             if factor * factor > N_factoring:
#                 break

#     # If N is still greater than 1, it must be a prime factor
#     if N_factoring > 1:
#         factors.append(N_factoring)

#     # Reverse the first half of the factors list if the first element is even
#     if factors[0] % 2 == 0:
#         for i in range(len(factors) // 2):
#             factors[i], factors[len(factors) - i - 1] = factors[len(factors) - i - 1], factors[i]

#     return factors

# fn _bit_reversed_range(N: Int) -> List[Int]:
#     var result = List[Int](capacity=N)

#     var bits = bit_width(N) - 1
#     for i in range(N):
#         var reversed_i = 0
#         for j in range(bits):
#             reversed_i = (reversed_i << 1) | ((i >> j) & 1)

#         result.append(reversed_i)

#     return result

# fn _twiddle_factors(N: Int) -> NumericArray[DType.float64, complex=True]:
#     """
#     Initializes Twiddle factors (complex roots of unity)
#     """
#     var factors = NumericArray[DType.float64, complex=True](count=N)

#     try:
#         var factor_term = -2 * pi / N

#         if N % 2 == 0:
#             for k in range(N // 2):
#                 var k_factor_term = k * factor_term
#                 var factor = ScalarNumber[DType.float64, complex=True](real=cos(k_factor_term), imaginary=sin(k_factor_term))

#                 factors[k] = factor
#                 factors[k + N // 2] = -factor
#         else:
#             for k in range(N):
#                 var k_factor_term = k * factor_term

#                 factors[k] = ScalarNumber[DType.float64, complex=True](real=cos(k_factor_term), imaginary=sin(k_factor_term))

#     except error:
#         fatal_error(error)

#     return factors^


# @parameter
# fn copy_as_complex_to_result[width: Int](value: Number[dtype, width, complex=False], row: Int, col: Int, component: Int):
#     try:
#         result.strided_store(
#             Number[DType.float64, width, complex=True](real=value.value.cast[DType.float64](), imaginary=0),
#             row=row,
#             col=col,
#             component=component
#         )
#     except error:
#         fatal_error(error)

# matrix.strided_iterate[copy_as_complex_to_result]()
