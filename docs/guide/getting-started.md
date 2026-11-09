# Getting Started

## Installation

Mosaic is installed with [Pixi](https://pixi.sh), the Mojo environment and package manager.

Add the Modular channels to `pixi.toml`:

```toml
[workspace]
channels = [
  "https://conda.modular.com/max",
  "https://repo.prefix.dev/modular-community",
  "conda-forge",
]
```

Then add Mosaic:

```bash
pixi add mosaic-mojo
```

## Quickstart

Load, process and show an image:

```mojo
from mosaic.image import Image, ImageFile, ColorSpace, Border, Interpolation
from mosaic.visualizer import Visualizer

def main() raises:
    # Load an RGB image as float64 (color space and data type are compile-time parameters)
    var image = Image[ColorSpace.rgb, DType.float64]("examples/data/mandrill.png")

    # Scale to double size with bilinear interpolation
    var scaled = image.scaled[Interpolation.bilinear](2.0)

    # Blur with a 9x9 Gaussian kernel and reflected borders
    scaled.gaussian_blur[Border.reflect](size=9)

    # Convert to 8-bit greyscale and save as a PNG
    var result = scaled.converted_as_type[ColorSpace.greyscale, DType.uint8]()
    result.save[ImageFile.png]("mandrill-processed")

    # Show the result in the native visualizer (macOS)
    Visualizer.show(result, window_title="Processed")
    Visualizer.wait()
```

## On the GPU

`DeviceImage` holds an image on the GPU. Operations are enqueued, and `to_host()` waits for them and returns an `Image`:

```mojo
from max.gpu.host import DeviceContext
from mosaic.image import Image, DeviceImage, ColorSpace, Border

def main() raises:
    var image = Image[ColorSpace.rgb, DType.uint8]("examples/data/mandrill.png")

    var filtered = (
        DeviceImage(image, DeviceContext())
        .bilateral_filtered[Border.reflect](radius=7, color_std_dev=80, space_std_dev=80)
        .to_host()
    )
```
