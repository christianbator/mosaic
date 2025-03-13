#
# test_image_construction.mojo
# mosaic
#
# Created by Christian Bator on 03/12/2025
#

from testing import assert_equal

from mosaic.matrix import Matrix
from mosaic.image import Image, ColorSpace

#
# Basic tests
#
fn test_image_construction() raises:
    var matrix = Matrix[DType.uint8, ColorSpace.greyscale.channels()].ascending(rows = 256, cols = 256)

    var image = Image[DType.uint8, ColorSpace.greyscale](matrix^)

    assert_equal(image[127, 127, 0], 127)

#
# Main
#
fn main() raises:
    test_image_construction()
