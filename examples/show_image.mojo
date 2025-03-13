#
# show_image.mojo
# mosaic
#
# Created by Christian Bator on 03/11/2025
#
from mosaic.matrix import Matrix
from mosaic.image import Image, ColorSpace
from mosaic.visualizer import Visualizer

fn main():
    var matrix = Matrix[DType.uint8, ColorSpace.greyscale.channels()].ascending(rows = 256, cols = 256)

    var blah = Image[DType.uint8, ColorSpace.greyscale](matrix^)
    print(blah[127, 127, 0])

    try:
        # Step 1: Load the image from a path and convert it to the desired  data type and color space
        var image = Image[DType.uint8, ColorSpace.rgb]("data/mandrill.png")

        # Step 2: Request the Visualizer to show the image in a window titled "Image"
        Visualizer.show(image = image, window_title = "Image")

        # Step 3: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
