#
# show_image.mojo
# mosaic
#
# Created by Christian Bator on 03/11/2025
#

from mosaic.image import Image, ColorSpace
from mosaic.visualizer import Visualizer

fn main():
    try:
        # Step 1: Load the image from a path and convert it to the desired  data type and color space
        var image = Image[DType.uint8, ColorSpace.rgb]("data/mandrill.png")

        # Step 2: Request the Visualizer to show the image in a window titled "Image"
        Visualizer.show(image = image, window_title = "Image")

        # Step 3: Wait for user interaction (CMD+W to closes the window)
        Visualizer.wait()

    except error:
        print(error)
