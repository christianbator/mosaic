#
# slice_image.mojo
# mosaic
#
# Created by Christian Bator on 03/16/2025
#

from mosaic.image import Image, ColorSpace, ImageSlice
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load the image from a path
        var image = Image[DType.uint8, ColorSpace.rgb]("data/mandrill.png")

        # Step 2: Create an ImageSlice with y values in [0: height / 2) and x values in [0, width)
        var top_half = image[: image.height() // 2, :]

        # Step 3: Request the Visualizer to show the image in a window titled "Image"
        Visualizer.show(top_half, window_title="Image")

        # Step 4: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
