#
# bilateral_filter.mojo
# mosaic
#
# Created by Christian Bator on 11/08/2026
#

from std.pathlib import Path
from std.time import perf_counter_ns
from max.gpu.host import DeviceContext

from mosaic.image import Image, DeviceImage, ColorSpace, Border, ImageFile

#
# Configuration
#
comptime input_dir = "benchmark/input"
comptime output_dir = "benchmark/output"
comptime results_path = "benchmark/output/bilateral_filter_mosaic.csv"

comptime radius = 7  # OpenCV d = 15
comptime sigma_color = 80.0
comptime sigma_space = 80.0

comptime warmup_iterations = 1
comptime iterations = 5

# OpenCV's default border, BORDER_REFLECT_101
comptime border = Border.reflect

comptime HostImage = Image[ColorSpace.rgb, DType.uint8]
comptime GPUImage = DeviceImage[ColorSpace.rgb, DType.uint8]


#
# Inputs
#
def input_paths() raises -> List[Path]:
    # Inputs written by prepare_input.py, named <source>_<width>x<height>.png
    var names = List[String]()

    for path in Path(input_dir).listdir():
        var name = path.name()

        if name.endswith(".png"):
            names.append(name)

    # Shorter names have fewer width digits, so this orders the inputs from smallest to largest
    def by_size(lhs: String, rhs: String) {} -> Bool:
        return lhs.byte_length() < rhs.byte_length() or (lhs.byte_length() == rhs.byte_length() and lhs < rhs)

    sort(names, by_size)

    var paths = List[Path]()

    for name in names:
        paths.append(Path(input_dir) / name)

    return paths^


def reference_path(input_path: Path) -> Path:
    return Path(output_dir) / (input_path.name().removesuffix(".png") + "_bilateral_filter_opencv.png")


def output_path(input_path: Path) -> Path:
    return Path(output_dir) / (input_path.name().removesuffix(".png") + "_bilateral_filter_mosaic.png")


#
# Measurement
#
def measure[Operation: def() raises -> None, //](operation: Operation) raises -> Float64:
    for _ in range(warmup_iterations):
        operation()

    var durations = List[Int]()

    for _ in range(iterations):
        var start = perf_counter_ns()
        operation()
        durations.append(Int(perf_counter_ns() - start))

    return median_milliseconds(durations^)


def median_milliseconds(var durations: List[Int]) -> Float64:
    sort(durations)

    return Float64(durations[len(durations) // 2]) / 1e6


def max_difference(image: HostImage, reference: HostImage) -> Int:
    var result = 0
    var image_data = image.unsafe_data_ptr()
    var reference_data = reference.unsafe_data_ptr()

    for i in range(image.samples()):
        var difference = abs(Int(image_data[unsafe_offset=i]) - Int(reference_data[unsafe_offset=i]))
        result = max(result, difference)

    return result


#
# Results
#
def aligned[T: Writable](value: T, width: Int) -> String:
    # Right-aligned in a column of the given width, or left-aligned for a negative width
    var text = String(value)
    var padding = " " * max(abs(width) - text.byte_length(), 0)

    return text + padding if width < 0 else padding + text


def report(mut rows: List[String], mode: String, image: HostImage, milliseconds: Float64, max_difference: Int):
    var megapixels = Float64(image.height() * image.width()) / 1e6
    var size = String(image.width(), "x", image.height())

    print(
        aligned(size, 12),
        aligned(round(megapixels, 2), 6),
        " ",
        aligned(mode, -22),
        aligned(round(milliseconds, 1), 9),
        aligned(round(megapixels / milliseconds * 1e3, 1), 8),
        aligned(max_difference, 9),
    )

    rows.append(String("mosaic,", mode, ",", image.width(), ",", image.height(), ",", megapixels, ",", milliseconds, ",", max_difference))


def write_results(rows: List[String]) raises:
    with open(results_path, "w") as file:
        file.write("library,mode,width,height,megapixels,milliseconds,max_difference\n")

        for row in rows:
            file.write(row, "\n")

    print("Results written to", results_path)


#
# Main
#
def main() raises:
    var context = DeviceContext()

    print("mosaic | GPU: ", context.name(), " (", context.api(), ")", sep="")
    print("bilateral_filtered(radius=", radius, ", color_std_dev=", sigma_color, ", space_std_dev=", sigma_space, "), median of ", iterations, " runs\n", sep="")
    print(aligned("size", 12), aligned("MP", 6), " ", aligned("mode", -22), aligned("ms", 9), aligned("MP/s", 8), aligned("max diff", 9))

    var rows = List[String]()

    for path in input_paths():
        var image = HostImage(path)
        var reference = HostImage(reference_path(path))

        # GPU end to end: host image in, host image out
        def filter_end_to_end() raises {imm image, imm context}:
            _ = GPUImage(image, context).bilateral_filtered[border](radius, sigma_color, sigma_space).to_host()

        var gpu_result = GPUImage(image, context).bilateral_filtered[border](radius, sigma_color, sigma_space).to_host()
        var gpu_difference = max_difference(gpu_result, reference)
        report(rows, String(context.api(), " (end-to-end)"), image, measure(filter_end_to_end), gpu_difference)

        # GPU resident: input already on the device, waiting for the result without downloading it
        var device_image = GPUImage(image, context)

        def filter_resident() raises {imm device_image}:
            var result = device_image.bilateral_filtered[border](radius, sigma_color, sigma_space)
            result.synchronize()

        report(rows, String(context.api(), " (resident)"), image, measure(filter_resident), gpu_difference)

        gpu_result.save[ImageFile.png](output_path(path))

        print()

    write_results(rows)
