#
# extract_channel.mojo
# mosaic
#
# Created by Christian Bator on 04/24/2025
#

from mosaic.image import Image, ColorSpace
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load an image from a filepath, specifying the desired data type and color space
        var image = Image[DType.uint8, ColorSpace.rgb]("data/squirrel.jpeg")

        # Step 2: Extract the green channel from the image
        var green = image.extract_channel[1]()

        # Step 3: Create an RGB image with only green data (channel 1)
        var green_image = Image[DType.uint8, ColorSpace.rgb].with_single_channel_data[1](green)

        # Step 4: Request the Visualizer to show the image in a window titled "Green Channel"
        Visualizer.show(green_image, window_title="Green Channel")

        # Step 5: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
