#
# resize_image.mojo
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

        # Step 2: Resize the image by stretching, optionally specifying an interpolation type
        var resized = image.resized[Interpolation.bilinear](height=256, width=512)

        # Step 3: Request the Visualizer to show the resized image in a window titled "Resized"
        Visualizer.show(resized, window_title="Resized")

        # Step 4: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
