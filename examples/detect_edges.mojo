#
# detect_edges.mojo
# mosaic
#
# Created by Christian Bator on 04/23/2025
#

from mosaic.image import Image, ColorSpace, Border
from mosaic.numeric import Matrix, ScalarNumber
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load an image from a filepath as float64 in greyscale for processing
        var image = Image[DType.float64, ColorSpace.greyscale]("data/mandrill.png")

        # Step 2: Smooth the image for better edge detection
        image.gaussian_blur[Border.reflect](7)

        # Step 3: Define a custom edge detection kernel with the same data type and depth for the color space (e.g. Laplacian).
        #         For multi-channel color spaces like RGB, filters are applied to each channel separately
        #         (see `Matrix.strided_replication()` to duplicate a filter kernel across channels).
        var kernel = Matrix[DType.float64, ColorSpace.greyscale.channels()](
            rows=3,
            cols=3,
            values=List[ScalarNumber[DType.float64]](
                0,  1, 0,
                1, -4, 1,
                0,  1, 0
            )
        )

        # Step 4: Filter the image with reflected border handling and the custom kernel
        var edges = image.filtered[Border.reflect](kernel)

        # Step 5: Request the Visualizer to show the edges in a window titled "Edges"
        Visualizer.show(edges, window_title="Edges")

        # Step 6: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
