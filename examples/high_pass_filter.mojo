#
# high_pass_spectral_filter.mojo
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

        # Step 3: Copy the spectrum to a center-shifted "high pass" spectrum for processing
        var high_pass_spectrum = spectrum.shifted_origin_to_center()

        # Step 4: Zero out the center of the high pass spectrum to discard low frequencies
        alias filter_size = 64

        var start_row = (spectrum.rows() - filter_size) // 2
        var start_col = (spectrum.cols() - filter_size) // 2

        var low_frequency_slice = high_pass_spectrum[
            start_row : start_row + filter_size,
            start_col : start_col + filter_size
        ]

        low_frequency_slice.fill(0)

        # Step 5: Create a high pass spectral image for visualization (see `fourier_transform.mojo` example for details)
        var high_pass_spectral_image = Image[DType.uint8, ColorSpace.greyscale](
            (high_pass_spectrum.norm() + 1).log().mapped_to_range(0, 255).astype[DType.uint8]()
        )

        # Step 6: Create the high pass image from the filtered spectrum, mapping to the uint8 range
        var high_pass_filtered_image = Image[DType.uint8, ColorSpace.greyscale].from_spectrum(
            high_pass_spectrum.shifted_center_to_origin(), lower_bound=0, upper_bound=255
        )

        # Step 7: Horizontally stack the image, spectral image, and filtered image for visualization
        var stacked = image.horizontally_stacked(high_pass_spectral_image).horizontally_stacked(high_pass_filtered_image)

        # Step 8: Request the Visualizer to show the stacked images in a window titled "High Pass Filter"
        Visualizer.show(stacked, window_title="High Pass Filter")

        # Step 9: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
