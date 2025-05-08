#
# rotate_image.mojo
# mosaic
#
# Created by Christian Bator on 04/23/2025
#

from mosaic.image import Image, ColorSpace
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load an image from a filepath, specifying the desired color space and data type
        var image = Image[ColorSpace.rgb, DType.uint8]("data/mandrill.png")

        # Step 2: Rotate the image 90°, specifying a direction
        var rotated = image.rotated_90[clockwise=True]()

        # Step 3: Request the Visualizer to show the rotated image in a window titled "Rotated"
        Visualizer.show(rotated, window_title="Rotated")

        # Step 4: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
