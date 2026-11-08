#
# prepare_input.py
# mosaic
#
# Created by Christian Bator on 11/08/2026
#

from pathlib import Path

import cv2
import numpy as np

#
# Configuration
#
SOURCE_PATHS = [Path("benchmark/data/macaw.jpeg")]
INPUT_DIR = Path("benchmark/input")

# Doubling sizes for an evenly spaced log-scale plot, None is the full-size image
MEGAPIXELS = [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, None]


#
# Resizing
#
def resized(image: np.ndarray, megapixels: float | None) -> np.ndarray:
    if megapixels is None:
        return image

    height, width = image.shape[:2]
    scale = (megapixels * 1e6 / (height * width)) ** 0.5

    # Area interpolation averages every source pixel, which avoids aliasing when downscaling
    return cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)


#
# Main
#
def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)

    for source_path in SOURCE_PATHS:
        # Decode once, then write lossless PNGs so every benchmark reads identical pixels
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)

        if source is None:
            raise RuntimeError(f"Failed to read image: {source_path} (run `pixi run download-benchmark-data`)")

        for megapixels in MEGAPIXELS:
            image = resized(source, megapixels)
            height, width = image.shape[:2]
            path = INPUT_DIR / f"{source_path.stem}_{width}x{height}.png"

            cv2.imwrite(str(path), image)
            print(f"> Wrote {path}")


if __name__ == "__main__":
    main()
