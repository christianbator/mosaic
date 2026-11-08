#
# bilateral_filter.py
# mosaic
#
# Created by Christian Bator on 11/08/2026
#

import csv
import statistics
import time
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

#
# Configuration
#
INPUT_DIR = Path("benchmark/input")
OUTPUT_DIR = Path("benchmark/output")
RESULTS_PATH = OUTPUT_DIR / "bilateral_filter_opencv.csv"

DIAMETER = 15
SIGMA_COLOR = 80.0
SIGMA_SPACE = 80.0

WARMUP_ITERATIONS = 1
ITERATIONS = 5


#
# Inputs
#
def input_paths() -> list[Path]:
    # Inputs written by prepare_input.py, named <source>_<width>x<height>.png
    paths = sorted(INPUT_DIR.glob("*.png"), key=pixel_count)

    if not paths:
        raise RuntimeError(f"No inputs found in {INPUT_DIR} (run `pixi run prepare-benchmark-input`)")

    return paths


def pixel_count(path: Path) -> int:
    width, height = path.stem.rsplit("_", 1)[1].split("x")

    return int(width) * int(height)


def reference_path(input_path: Path) -> Path:
    return OUTPUT_DIR / f"{input_path.stem}_bilateral_filter_opencv.png"


#
# Filters
#
def bilateral_filter(image: np.ndarray | cv2.UMat) -> np.ndarray | cv2.UMat:
    return cv2.bilateralFilter(image, DIAMETER, SIGMA_COLOR, SIGMA_SPACE)


#
# Measurement
#
def measure(operation: Callable[[], object]) -> float:
    for _ in range(WARMUP_ITERATIONS):
        operation()

    durations = []

    for _ in range(ITERATIONS):
        start = time.perf_counter_ns()
        operation()
        durations.append(time.perf_counter_ns() - start)

    return statistics.median(durations) / 1e6


def run_on_cpu(image: np.ndarray, threads: int) -> tuple[float, np.ndarray]:
    cv2.ocl.setUseOpenCL(False)
    cv2.setNumThreads(threads)

    duration = measure(lambda: bilateral_filter(image))

    return duration, bilateral_filter(image)


def run_on_opencl(image: np.ndarray) -> tuple[float, float, np.ndarray]:
    cv2.ocl.setUseOpenCL(True)

    # End-to-end: host image in, host image out
    end_to_end_duration = measure(lambda: bilateral_filter(cv2.UMat(image)).get())

    # Resident: input already on the device, wait for completion without downloading
    device_image = cv2.UMat(image)
    resident_duration = measure(lambda: (bilateral_filter(device_image), cv2.ocl.finish()))

    return end_to_end_duration, resident_duration, bilateral_filter(cv2.UMat(image)).get()


#
# Reporting
#
def print_header(default_threads: int):
    print(f"OpenCV {cv2.__version__} | CPU threads: {default_threads}", end="")

    if cv2.ocl.haveOpenCL():
        device = cv2.ocl.Device.getDefault()
        print(f" | OpenCL: {device.name()} ({device.OpenCLVersion()})")
    else:
        print(" | OpenCL: unavailable")

    print(f"bilateralFilter(d={DIAMETER}, sigmaColor={SIGMA_COLOR}, sigmaSpace={SIGMA_SPACE}), median of {ITERATIONS} runs\n")
    print(f"{'size':>12} {'MP':>6}  {'mode':<22} {'ms':>9} {'MP/s':>8} {'max diff':>9}")


def report(results: list[dict], image: np.ndarray, mode: str, duration: float, max_difference: int | None = None):
    height, width = image.shape[:2]
    megapixels = height * width / 1e6
    difference = "" if max_difference is None else str(max_difference)

    print(f"{f'{width}x{height}':>12} {megapixels:>6.2f}  {mode:<22} {duration:>9.1f} {megapixels / duration * 1e3:>8.1f} {difference:>9}")

    results.append(
        {
            "library": "opencv",
            "mode": mode,
            "width": width,
            "height": height,
            "megapixels": megapixels,
            "milliseconds": duration,
            "max_difference": difference,
        }
    )


def write_results(results: list[dict]):
    with open(RESULTS_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results written to {RESULTS_PATH}")


#
# Main
#
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    default_threads = cv2.getNumThreads()
    print_header(default_threads)

    results = []

    for path in input_paths():
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)

        cpu_duration, cpu_result = run_on_cpu(image, threads=default_threads)
        report(results, image, f"cpu ({default_threads} threads)", cpu_duration)

        single_thread_duration, _ = run_on_cpu(image, threads=1)
        report(results, image, "cpu (1 thread)", single_thread_duration)

        cv2.setNumThreads(default_threads)

        if cv2.ocl.haveOpenCL():
            end_to_end_duration, resident_duration, opencl_result = run_on_opencl(image)
            max_difference = int(np.max(cv2.absdiff(opencl_result, cpu_result)))
            report(results, image, "opencl (end-to-end)", end_to_end_duration, max_difference)
            report(results, image, "opencl (resident)", resident_duration, max_difference)

        # Reference output, for comparing against mosaic
        cv2.imwrite(str(reference_path(path)), cpu_result)

        print()

    write_results(results)


if __name__ == "__main__":
    main()
