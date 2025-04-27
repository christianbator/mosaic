#
# inverse_fourier_transform.mojo
# mosaic
#
# Created by Christian Bator on 04/24/2025
#

from mosaic.image import Image, ColorSpace
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load an image from a filepath, specifying the desired data type and color space
        var image = Image[DType.uint8, ColorSpace.greyscale]("data/camera.png")

        # Step 2: Calculate the image spectrum using the Fourier transform
        var spectrum = image.spectrum()

        # Step 3: Recreate the image from the spectrum, mapping to the uint8 range
        var recreated_image = Image[DType.uint8, ColorSpace.greyscale].from_spectrum(spectrum, lower_bound=0, upper_bound=255)

        # Step 4: Create a spectral image for visualization (see `fourier_transform.mojo` example for details)
        var spectral_image = Image[DType.uint8, ColorSpace.greyscale](
            (spectrum.shifted_origin_to_center().norm() + 1).log().mapped_to_range(0, 255).astype[DType.uint8]()
        )

        # Step 5: Horizontally stack the spectral image and recreated image for visualization
        var stacked = spectral_image.horizontally_stacked(recreated_image)

        # Step 6: Request the Visualizer to show the stacked images in a window titled "Recreated from Spectrum"
        Visualizer.show(stacked, window_title="Recreated from Spectrum")

        # Step 7: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
