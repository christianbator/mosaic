#
# fourier_transform.mojo
# mosaic
#
# Created by Christian Bator on 04/23/2025
#

from mosaic.image import Image, ColorSpace
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load an image from a filepath, specifying the desired color space and data type
        var image = Image[ColorSpace.greyscale, DType.uint8]("data/camera.png")

        # Step 2: Calculate the image spectrum using the Fourier transform
        var spectrum = image.spectrum()

        # Step 3: Shift the top left origin to the center for more intuitive viewing
        spectrum.shift_origin_to_center()

        # Step 4: Apply the log scale to the magnitude of the spectrum
        var scaled_spectrum = (spectrum.norm() + 1).log()

        # Step 5: Normalize the scaled spectrum by mapping to the uint8 range
        var normalized_spectrum = scaled_spectrum.mapped_to_range(0, 255).as_type[DType.uint8]()

        # Step 6: Create an image from the resulting spectral matrix
        var spectral_image = Image[ColorSpace.greyscale, DType.uint8](normalized_spectrum^)

        # Step 7: Horizontally stack the original image and its spectrum for visualization
        var stacked = image.horizontally_stacked(spectral_image)

        # Step 8: Request the Visualizer to show the stacked images in a window titled "Spectrum"
        Visualizer.show(stacked, window_title="Spectrum")

        # Step 9: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
