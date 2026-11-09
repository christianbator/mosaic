#
# test_device_image.mojo
# mosaic
#
# Created by Christian Bator on 11/08/2026
#

from std.sys import has_accelerator
from std.testing import assert_true, TestSuite
from max.gpu.host import DeviceContext

from mosaic.image import Image, DeviceImage, ColorSpace, Border

comptime input = "tests/data/input/"
comptime expected = "tests/data/output/"


def test_round_trip() raises:
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var result = DeviceImage(image, DeviceContext()).to_host()

    assert_true(result.matches(image, tolerance=0))


def test_device_memory() raises:
    # Device memory, as on a discrete GPU, instead of host memory on unified memory
    var context = DeviceContext()
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var device_image = DeviceImage[ColorSpace.rgb, DType.uint8](
        image.height(), image.width(), context, _host_memory=False
    )
    DeviceImage(image, context).copy_into(device_image)
    var result = device_image.to_host()

    assert_true(result.matches(image, tolerance=0))


def test_invert() raises:
    var context = DeviceContext()
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var reference = image.copy()
    reference.invert()

    var device_image = DeviceImage(image, context)
    assert_true(device_image.inverted().to_host().matches(reference, tolerance=0))

    device_image.invert()
    assert_true(device_image.to_host().matches(reference, tolerance=0))


def test_bilateral_filter() raises:
    # Reference from OpenCV: cv2.bilateralFilter(image, d=15, sigmaColor=80, sigmaSpace=80), which reflects at the border
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var cpu_result = image.bilateral_filtered[Border.reflect](radius=7, color_std_dev=80, space_std_dev=80)
    var gpu_result = (
        DeviceImage(image, DeviceContext())
        .bilateral_filtered[Border.reflect](radius=7, color_std_dev=80, space_std_dev=80)
        .to_host()
    )

    var reference = Image[ColorSpace.rgb, DType.uint8](expected + "bilateral_filter.png")
    assert_true(cpu_result.matches(reference))
    assert_true(gpu_result.matches(cpu_result, tolerance=0))


def test_bilateral_filter_edges() raises:
    # A width that isn't a multiple of the pixels each GPU thread filters, a single channel, and a zero border
    var image = Image[ColorSpace.greyscale, DType.uint8](input + "camera.png")[:, :250].deep_copy()
    var cpu_result = image.bilateral_filtered[Border.zero](radius=4, color_std_dev=30, space_std_dev=4)
    var gpu_result = (
        DeviceImage(image, DeviceContext())
        .bilateral_filtered[Border.zero](radius=4, color_std_dev=30, space_std_dev=4)
        .to_host()
    )

    assert_true(gpu_result.matches(cpu_result, tolerance=0))


def test_bilateral_filter_in_place() raises:
    var image = Image[ColorSpace.rgb, DType.uint8](input + "mandrill.png")
    var device_image = DeviceImage(image, DeviceContext())
    var reference = device_image.bilateral_filtered[Border.wrap](radius=3, color_std_dev=50, space_std_dev=10).to_host()

    device_image.bilateral_filter[Border.wrap](radius=3, color_std_dev=50, space_std_dev=10)
    assert_true(device_image.to_host().matches(reference, tolerance=0))


def main() raises:
    # Checked at compile time, so machines without a GPU never compile the GPU kernels
    comptime if has_accelerator():
        TestSuite.discover_tests[__functions_in_module()]().run()
    else:
        print("No GPU found, skipping DeviceImage tests")
