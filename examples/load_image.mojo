#
# load_image.mojo
# mosaic
#
# Created by Christian Bator on 03/11/2025
#

from mosaic.image import Image, ColorSpace
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load an image from a filepath, specifying the desired color space and data type
        var image = Image[ColorSpace.rgb, DType.uint8]("data/mandrill.png")

        # Step 2: Request the Visualizer to show the image in a window titled "Image"
        Visualizer.show(image, window_title="Image")

        # Step 3: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
