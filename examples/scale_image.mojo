#
# scale_image.mojo
# mosaic
#
# Created by Christian Bator on 04/23/2025
#

from mosaic.image import Image, ColorSpace, Interpolation
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load an image from a filepath, specifying the desired color space and data type
        var image = Image[ColorSpace.rgb, DType.uint8]("data/mandrill.png")

        # Step 2: Scale the image by a factor of 0.5, optionally specifying an interpolation type
        var scaled = image.scaled[interpolation = Interpolation.bilinear](0.5)

        # Step 3: Request the Visualizer to show the scaled image in a window titled "Scaled"
        Visualizer.show(scaled, window_title="Scaled")

        # Step 4: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
