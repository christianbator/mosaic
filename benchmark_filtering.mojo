#
# benchmark_filtering.mojo
# mosaic
#
# Created by Christian Bator on 03/11/2025
#

import benchmark
from benchmark import Unit

from mosaic.matrix import Matrix
from mosaic.image import Image, ImageFileType, ColorSpace, Border
from mosaic.visualizer import Visualizer

fn main():
    try:
        alias color_space = ColorSpace.rgb
        alias dtype = DType.float32
        alias kernel_size = 9

        var original = Image[dtype, color_space]("data/mandrill.png")

        var kernel = Matrix[dtype, color_space.channels()](
            rows = kernel_size,
            cols = kernel_size,
            number = 1.0 / Scalar[dtype](kernel_size**2)
        )

        @parameter
        fn filter():
            _ = original.filtered[Border.zero](kernel)
        
        var direct_report = benchmark.run[filter](max_iters = 100, max_runtime_secs = 10)
        print("Filter: ", direct_report.mean(Unit.ms), "ms", "(" + String(direct_report.iters()) + " iterations)")

        var filtered = original.filtered[Border.zero](kernel)
        Visualizer.show(image = filtered, window_title = "Filtered")
        Visualizer.wait()

    except error:
        print(error)
