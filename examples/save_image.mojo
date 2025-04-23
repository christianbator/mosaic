#
# save_image.mojo
# mosaic
#
# Created by Christian Bator on 04/23/2025
#

from mosaic.image import Image, ImageFile, ColorSpace


fn main():
    try:
        # Step 1: Load an image from a filepath, specifying the desired data type and color space
        var image = Image[DType.uint8, ColorSpace.rgb]("data/mandrill.png")

        # Step 2: Save a copy of the image to a new location, specifying the image file type
        image.save[ImageFile.png]("copy")

    except error:
        print(error)
