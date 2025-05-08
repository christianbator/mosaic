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
        # Step 1: Load an image from a filepath, specifying the desired color space and data type
        var image = Image[ColorSpace.rgb, DType.uint8]("data/mandrill.png")

        # Step 2: Pad the image with a specified height and width, optionally specifying a border handling method
        var padded = image.padded[Border.zero](height=44, width=44)

        # Step 3: Request the Visualizer to show the padded image in a window titled "Padded"
        Visualizer.show(padded, window_title="Padded")

        # Step 4: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
