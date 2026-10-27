#
# test_examples.mojo
# mosaic
#
# Created by Christian Bator on 03/01/2026
#

from std.testing import assert_true, assert_equal, TestSuite

from mosaic.image import Image, ImageFile, ColorSpace, Border, Interpolation
from mosaic.numeric import Matrix

comptime input = "tests/data/input/"
comptime expected = "tests/data/output/"


def test_blur_image() raises:
    var image = Image[ColorSpace.rgb, DType.float64](input + "mandrill.png")
    image.gaussian_blur[Border.reflect](size=9)
    var result = image.as_type[DType.uint8]()

    var reference = Image[ColorSpace.rgb, DType.uint8](expected + "blur_image.png")
    assert_true(result.matches(reference))


def test_convert_as_type() raises:
    var image = Image[ColorSpace.rgb, DType.float64](input + "mandrill.png")
    var result = image.converted_as_type[ColorSpace.greyscale, DType.uint8]()

    var reference = Image[ColorSpace.greyscale, DType.uint8](expected + "convert_as_type.png")
    assert_true(result.matches(reference))


def test_convert_color_space() raises:
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var result = image.converted[ColorSpace.greyscale]()

    var reference = Image[ColorSpace.greyscale, DType.uint8](expected + "convert_color_space.png")
    assert_true(result.matches(reference))


def test_convert_type() raises:
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var result = image.as_type[DType.float64]().as_type[DType.uint8]()

    var reference = Image[ColorSpace.rgb, DType.uint8](expected + "convert_type.png")
    assert_true(result.matches(reference))


def test_detect_edges() raises:
    var image = Image[ColorSpace.greyscale, DType.float64](input + "mandrill.png")
    image.gaussian_blur[Border.reflect](7)

    var kernel = Matrix[DType.float64, ColorSpace.greyscale.channels()](
        rows=3,
        cols=3,
        values=[0, 1, 0, 1, -4, 1, 0, 1, 0],
    )
    # Convert through int64 so negative edge values wrap deterministically (float -> uint8 is undefined when out of range)
    var result = image.filtered[Border.reflect](kernel).as_type[DType.int64]().as_type[DType.uint8]()

    var reference = Image[ColorSpace.greyscale, DType.uint8](expected + "detect_edges.png")
    assert_true(result.matches(reference))


def test_extract_channel() raises:
    var image = Image[ColorSpace.rgb, DType.uint8](input + "squirrel.jpeg")
    var green = image.extract_channel[1]()
    var result = Image[ColorSpace.rgb, DType.uint8].with_single_channel_data[1](green)

    var reference = Image[ColorSpace.rgb, DType.uint8](expected + "extract_channel.png")
    assert_true(result.matches(reference))


def test_flip_image() raises:
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var result = image.flipped_vertically()

    var reference = Image[ColorSpace.rgb, DType.uint8](expected + "flip_image.png")
    assert_true(result.matches(reference))


def test_fourier_transform() raises:
    var image = Image[ColorSpace.greyscale, DType.uint8](input + "camera.png")
    var spectrum = image.spectrum()
    spectrum.shift_origin_to_center()
    var scaled_spectrum = (spectrum.norm() + 1).log()
    var norm_spectrum = scaled_spectrum.mapped_to_range(0, 255).as_type[DType.uint8]()
    var spec_img = Image[ColorSpace.greyscale, DType.uint8](norm_spectrum^)
    var result = image.horizontally_stacked(spec_img)

    var reference = Image[ColorSpace.greyscale, DType.uint8](expected + "fourier_transform.png")
    assert_true(result.matches(reference))


def test_high_pass_filter() raises:
    var image = Image[ColorSpace.greyscale, DType.uint8](input + "camera.png")
    var spectrum = image.spectrum()
    var hp_spectrum = spectrum.shifted_origin_to_center()

    comptime filter_size = 64
    var start_row = (spectrum.rows() - filter_size) // 2
    var start_col = (spectrum.cols() - filter_size) // 2
    var lf_slice = hp_spectrum[start_row : start_row + filter_size, start_col : start_col + filter_size]
    lf_slice.fill(0)

    var hp_spectral_image = Image[ColorSpace.greyscale, DType.uint8](
        (hp_spectrum.norm() + 1).log().mapped_to_range(0, 255).as_type[DType.uint8]()
    )
    var hp_filtered = Image[ColorSpace.greyscale, DType.uint8].from_spectrum(
        hp_spectrum.shifted_center_to_origin(), lower_bound=0, upper_bound=255
    )
    var result = image.horizontally_stacked(hp_spectral_image).horizontally_stacked(hp_filtered)

    var reference = Image[ColorSpace.greyscale, DType.uint8](expected + "high_pass_filter.png")
    assert_true(result.matches(reference))


def test_inverse_fourier_transform() raises:
    var image = Image[ColorSpace.greyscale, DType.uint8](input + "camera.png")
    var spectrum = image.spectrum()
    var recreated = Image[ColorSpace.greyscale, DType.uint8].from_spectrum(spectrum, lower_bound=0, upper_bound=255)
    var spectral_image = Image[ColorSpace.greyscale, DType.uint8](
        (spectrum.shifted_origin_to_center().norm() + 1).log().mapped_to_range(0, 255).as_type[DType.uint8]()
    )
    var result = spectral_image.horizontally_stacked(recreated)

    var reference = Image[ColorSpace.greyscale, DType.uint8](expected + "inverse_fourier_transform.png")
    assert_true(result.matches(reference))


def test_pad_image() raises:
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var result = image.padded[Border.zero](height=44, width=44)

    var reference = Image[ColorSpace.rgb, DType.uint8](expected + "pad_image.png")
    assert_true(result.matches(reference))


def test_picture_in_picture() raises:
    var image = Image[ColorSpace.rgb, DType.uint8](input + "squirrel.jpeg")

    comptime squirrel_head_size = 150
    var head = image[120 : 120 + squirrel_head_size, 240 : 240 + squirrel_head_size]
    var padded_head = head.deep_copy().padded(2)
    image.store_sub_image(padded_head, y=20, x=20)

    var reference = Image[ColorSpace.rgb, DType.uint8](expected + "picture_in_picture.png")
    assert_true(image.matches(reference))


def test_resize_image() raises:
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var result = image.resized[Interpolation.bilinear](height=256, width=512)

    var reference = Image[ColorSpace.rgb, DType.uint8](expected + "resize_image.png")
    assert_true(result.matches(reference))


def test_rotate_image() raises:
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var result = image.rotated_90[clockwise=True]()

    var reference = Image[ColorSpace.rgb, DType.uint8](expected + "rotate_image.png")
    assert_true(result.matches(reference))


def test_scale_image() raises:
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var result = image.scaled[interpolation=Interpolation.bilinear](0.5)

    var reference = Image[ColorSpace.rgb, DType.uint8](expected + "scale_image.png")
    assert_true(result.matches(reference))


def test_slice_image() raises:
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var result = image[: image.height() // 2, :].deep_copy()

    var reference = Image[ColorSpace.rgb, DType.uint8](expected + "slice_image.png")
    assert_true(result.matches(reference))


def test_unsharp_mask() raises:
    var original = Image[ColorSpace.rgb, DType.float64](input + "mandrill.png")
    var blurred = original.gaussian_blurred[Border.reflect](7)
    var mask = original - blurred
    var unsharp = original + (0.8 * mask)
    unsharp.clamp(0, 255)
    var result = original.as_type[DType.uint8]().horizontally_stacked(unsharp.as_type[DType.uint8]())

    var reference = Image[ColorSpace.rgb, DType.uint8](expected + "unsharp_mask.png")
    assert_true(result.matches(reference))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
