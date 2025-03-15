#
# filters.mojo
# mosaic
#
# Created by Christian Bator on 03/14/2025
#

from math import pi, exp
from collections import Optional

from mosaic.numeric import Matrix, Number, ScalarNumber


struct Filters:
    @staticmethod
    fn box_kernel_1d[
        dtype: DType, depth: Int = 1, complex: Bool = False
    ](size: Int) -> Matrix[dtype, depth, complex]:
        constrained[
            dtype.is_floating_point(),
            "box_kernel_1d() is only available for floating point dtypes",
        ]()

        return Matrix[dtype, depth, complex](
            rows=size, cols=1, value=(1 / size).cast[dtype]()
        )

    @staticmethod
    fn box_kernel_2d[
        dtype: DType, depth: Int = 1, complex: Bool = False
    ](size: Int) -> Matrix[dtype, depth, complex]:
        constrained[
            dtype.is_floating_point(),
            "box_kernel_2d() is only available for floating point dtypes",
        ]()

        return Matrix[dtype, depth, complex](
            rows=size, cols=size, value=(1 / (size * size)).cast[dtype]()
        )

    @staticmethod
    fn gaussian_kernel_1d[
        dtype: DType, depth: Int = 1, complex: Bool = False
    ](size: Int, std_dev: Optional[Float64] = None) -> Matrix[
        dtype, depth, complex
    ]:
        constrained[
            dtype.is_floating_point(),
            "gaussian_kernel_1d() is only available for floating point dtypes",
        ]()

        if size == 1:
            return Matrix[dtype, depth, complex].strided_replication(
                size, 1, 1.0
            )
        elif std_dev:
            var result = Matrix[DType.float64, depth, complex](
                rows=size, cols=1
            )
            var variance = std_dev.value() ** 2

            for i in range(size):
                result.store_full_depth(
                    row=i,
                    col=0,
                    value=result.create_full_depth_value(
                        exp(-((i - (size - 1) / 2) ** 2) / (2 * variance))
                    ),
                )

            result.strided_normalize()

            return result.astype[dtype]()
        else:
            if size == 3:
                return Matrix[dtype, depth, complex].strided_replication(
                    size, 1, 0.250, 0.500, 0.250
                )
            elif size == 5:
                return Matrix[dtype, depth, complex].strided_replication(
                    size, 1, 0.062500, 0.250000, 0.375000, 0.250000, 0.062500
                )
            elif size == 7:
                return Matrix[dtype, depth, complex].strided_replication(
                    size,
                    1,
                    0.031250,
                    0.109375,
                    0.218750,
                    0.281250,
                    0.218750,
                    0.109375,
                    0.031250,
                )
            elif size == 9:
                return Matrix[dtype, depth, complex].strided_replication(
                    size,
                    1,
                    0.015625000,
                    0.050781250,
                    0.117187500,
                    0.199218750,
                    0.234375000,
                    0.199218750,
                    0.117187500,
                    0.050781250,
                    0.015625000,
                )
            else:
                var result = Matrix[DType.float64, depth, complex](
                    rows=size, cols=1
                )
                var variance = (0.3 * ((size - 1) * 0.5 - 1.0) + 0.8) ** 2

                for i in range(size):
                    result.store_full_depth(
                        row=i,
                        col=0,
                        value=result.create_full_depth_value(
                            exp(-((i - (size - 1) / 2) ** 2) / (2 * variance))
                        ),
                    )

                result.strided_normalize()

                return result.astype[dtype]()

    @staticmethod
    fn gaussian_kernel_2d[
        dtype: DType, depth: Int = 1, complex: Bool = False
    ](size: Int, std_dev: Optional[Float64] = None) -> Matrix[
        dtype, depth, complex
    ]:
        constrained[
            dtype.is_floating_point(),
            "gaussian_kernel_2d() is only available for floating point dtypes",
        ]()

        var kernel = Self.gaussian_kernel_1d[dtype, depth, complex](
            size=size, std_dev=std_dev
        )

        return kernel @ kernel.transposed()
