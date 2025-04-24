#
# picture_in_picture.mojo
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

        # Step 2: Extract a sub-rect from the image using slice notation
        alias squirrel_head_size = 150

        var squirrel_head = image[120 : 120 + squirrel_head_size, 240 : 240 + squirrel_head_size]

        # Step 3: Materialize the slice into an Image with `copy()` and add a black border
        var padded_squirrel_head = squirrel_head.copy().padded(2)

        # Step 4: Store the sub-image in the original image in the top left corner
        image.store_sub_image(padded_squirrel_head, y=20, x=20)

        # Step 5: Request the Visualizer to show the image in a window titled "Picture in Picture"
        Visualizer.show(image, window_title="Picture in Picture")

        # Step 6: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
