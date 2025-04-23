#
# convert_color_space.mojo
# mosaic
#
# Created by Christian Bator on 04/23/2025
#

from mosaic.image import Image, ColorSpace
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load an image from a filepath, specifying the desired data type and color space
        var image = Image[DType.uint8, ColorSpace.rgb]("data/mandrill.png")

        # Step 2: Convert the image to greyscale
        var greyscale = image.converted[ColorSpace.greyscale]()

        # Step 3: Request the Visualizer to show the greyscale image in a window titled "Greyscale"
        Visualizer.show(greyscale, window_title="Greyscale")

        # Step 4: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
