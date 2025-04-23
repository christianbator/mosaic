#
# pad_image.mojo
# mosaic
#
# Created by Christian Bator on 04/23/2025
#

from mosaic.image import Image, ColorSpace, Border
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load an image from a filepath, specifying the desired data type and color space
        var image = Image[DType.uint8, ColorSpace.rgb]("data/mandrill.png")

        # Step 2: Pad the image with a specified width and height, optionally specifying a border handling method
        var padded = image.padded[Border.zero](width=44, height=44)

        # Step 3: Request the Visualizer to show the padded image in a window titled "Padded"
        Visualizer.show(padded, window_title="Padded")

        # Step 4: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
