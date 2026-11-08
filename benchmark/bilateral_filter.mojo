#
# bilateral_filter.mojo
# mosaic
#
# Created by Christian Bator on 11/08/2026
#

from std.pathlib import Path
from std.time import perf_counter_ns

from mosaic.image import Image, ColorSpace

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

comptime HostImage = Image[ColorSpace.rgb, DType.uint8]


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


#
# Measurement
#
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
def result_row(mode: String, image: HostImage, milliseconds: Float64, max_difference: Int) -> String:
    var megapixels = Float64(image.height() * image.width()) / 1e6

    return String("mosaic,", mode, ",", image.width(), ",", image.height(), ",", megapixels, ",", milliseconds, ",", max_difference)


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
    print("mosaic bilateral filter(radius=", radius, ", sigma_color=", sigma_color, ", sigma_space=", sigma_space, "), median of ", iterations, " runs\n", sep="")

    var rows = List[String]()

    for path in input_paths():
        var image = HostImage(path)
        var reference = HostImage(reference_path(path))

        print(image.width(), "x", image.height(), sep="")

        # TODO: Bilateral filter on the GPU
        #
        #   var durations = List[Int]()
        #
        #   for i in range(warmup_iterations + iterations):
        #       var start = perf_counter_ns()
        #       ... filter, then synchronize ...
        #       if i >= warmup_iterations:
        #           durations.append(Int(perf_counter_ns() - start))
        #
        #   rows.append(result_row("gpu (resident)", image, median_milliseconds(durations^), max_difference(result, reference)))

    if len(rows) > 0:
        write_results(rows)
