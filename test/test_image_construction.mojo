#
# test_image_construction.mojo
# mosaic
#
# Created by Christian Bator on 03/12/2025
#

from testing import assert_true

from mosaic.numeric import Matrix
from mosaic.image import Image, ColorSpace


fn test_image_matrix_construction() raises:
    var matrix = Matrix[DType.uint8, ColorSpace.greyscale.channels()].ascending(rows=256, cols=256)

    var image = Image[ColorSpace.greyscale, DType.uint8](matrix^)

    assert_true(image[127, 127] == 127)


fn test_image_png_construction() raises:
    var image = Image[ColorSpace.greyscale, DType.uint8]("data/mandrill.png")

    assert_true(image[120, 240] == 44)
