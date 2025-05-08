#
# unsharp_mask.mojo
# mosaic
#
# Created by Christian Bator on 04/23/2025
#

from mosaic.image import Image, ColorSpace, Border
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Load an image from a filepath as RGB float64 for processing
        var original = Image[ColorSpace.rgb, DType.float64]("data/mandrill.png")

        # Step 2: Create a blurred version of the image
        var blurred = original.gaussian_blurred[Border.reflect](7)

        # Step 3: Create a mask containing high-frequency details (difference between original and blurred)
        var mask = original - blurred

        # Step 4: Add a scaled version of the mask back to the original
        var unsharp_masked = original + (0.8 * mask)

        # Step 5: Clamp to the uint8 range
        unsharp_masked.clamp(0, 255)

        # Step 6: Horizontally stack the original and unsharp masked image to visualize the difference
        var stacked = original.horizontally_stacked(unsharp_masked)

        # Step 7: Request the Visualizer to show the stacked images in a window titled "Unsharp Masked"
        Visualizer.show(stacked, window_title="Unsharp Masked")

        # Step 8: Wait for user interaction (CMD+W closes the window)
        Visualizer.wait()

    except error:
        print(error)
