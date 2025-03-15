#
# blur_image.mojo
# mosaic
#
# Created by Christian Bator on 03/14/2025
#

from mosaic.image import Image, ColorSpace, Border
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load the image from a path and convert it to float32 for processing
        var image = Image[DType.float32, ColorSpace.rgb]("data/mandrill.png")

        # Step 2: Blur the image with zero-padded border handling and a box filter of size 9
        image.box_blur[Border.zero](size=9)

        # Step 3: Request the Visualizer to show the image in a window titled "Image"
        Visualizer.show(image=image, window_title="Image")

        # Step 4: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
