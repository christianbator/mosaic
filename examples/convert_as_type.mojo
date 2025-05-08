#
# convert_as_type.mojo
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
        var image = Image[ColorSpace.rgb, DType.float64]("data/mandrill.png")

        # Step 2: Convert the image to a new color space and data type in one method
        var greyscale_uint8 = image.converted_as_type[ColorSpace.greyscale, DType.uint8]()

        # Step 3: Request the Visualizer to show the greyscale uint8 image in a window titled "Greyscale UInt8"
        Visualizer.show(greyscale_uint8, window_title="Greyscale UInt8")

        # Step 4: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
