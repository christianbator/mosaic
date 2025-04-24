#
# convert_astype.mojo
# mosaic
#
# Created by Christian Bator on 04/23/2025
#

from mosaic.image import Image, ColorSpace
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load an image from a filepath into any data type and color space representation
        #         (the specified data type and color space don't have to match the image file).
        var image = Image[DType.float64, ColorSpace.rgb]("data/mandrill.png")

        # Step 2: Convert the image to a new color space and data type in one method
        var uint8_greyscale = image.converted_astype[DType.uint8, ColorSpace.greyscale]()

        # Step 3: Request the Visualizer to show the uint8 greyscale image in a window titled "UInt8 Greyscale"
        Visualizer.show(uint8_greyscale, window_title="UInt8 Greyscale")

        # Step 4: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
