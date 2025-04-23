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
        # Step 1: Load an image from a filepath as float64 for processing
        var image = Image[DType.float64, ColorSpace.rgb]("data/mandrill.png")

        # Step 2: Blur the image with reflected border handling and a gaussian filter of size 9
        image.gaussian_blur[Border.reflect](size=9)

        # Step 3: Request the Visualizer to show the image in a window titled "Blurred"
        Visualizer.show(image, window_title="Blurred")

        # Step 4: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
