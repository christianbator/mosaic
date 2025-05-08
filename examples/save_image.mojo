#
# save_image.mojo
# mosaic
#
# Created by Christian Bator on 04/23/2025
#

from mosaic.image import Image, ColorSpace, ImageFile


fn main():
    try:
        # Step 1: Load an image from a filepath, specifying the desired color space and data type
        var image = Image[ColorSpace.rgb, DType.uint8]("data/mandrill.png")

        # Step 2: Save a copy of the image to a new location, specifying the image file type
        image.save[ImageFile.png]("copy")

    except error:
        print(error)
