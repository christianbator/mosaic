#
# convert_type.mojo
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

        # Step 2: Convert the image to a floating point data type for processing
        var float_image = image.astype[DType.float64]()

        # Step 3: The Visualizer automatically converts images to a renderable data type
        Visualizer.show(float_image, window_title="Float Image")

        # Step 4: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
