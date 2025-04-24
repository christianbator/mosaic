#
# video_processing.mojo
# mosaic
#
# Created by Christian Bator on 04/24/2025
#

from mosaic.image import Image, ColorSpace, Border
from mosaic.video import VideoCapture, VideoCapturing
from mosaic.numeric import Matrix, ScalarNumber
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Attempt to start video capture for the first available camera in the specifed color space.
        #         You can also specify a device by its name (e.g. "MX Brio").
        var video_capture = VideoCapture[ColorSpace.rgb](0)

        # Step 2: Start the camera capture
        video_capture.start()

        # Step 3: Define a custom Laplacian kernel for edge detection
        var kernel = Matrix[DType.float64, ColorSpace.greyscale.channels()](
            rows = 3,
            cols = 3,
            values = List[ScalarNumber[DType.float64]](
                0,  1,  0,
                1, -4,  1,
                0,  1,  0
            )
        )

        # Step 4: Define a frame processor that is invoked for each new frame, taking a pointer to the image and returning a processed image
        @parameter
        fn process_frame[V: VideoCapturing](image: Pointer[Image[DType.uint8, V.color_space]]) -> Image[DType.uint8, ColorSpace.greyscale]:
            try:
                # Convert to float64 greyscale for processing
                var greyscale = image[].converted_astype[DType.float64, ColorSpace.greyscale]()

                # Smooth the image for better edge detection
                var blurred = greyscale.gaussian_blurred[Border.reflect](5)

                # Filter and scale output
                var edges = 15 * blurred.filtered[Border.reflect](kernel)

                # Clamp to uint8 range
                edges.clamp(0, 255)
                
                # Return processed image as renderable type
                return edges.astype[DType.uint8]()

            except error:
                print(error)
                return image[].converted[ColorSpace.greyscale]()

        # Step 5: Visualize the processed frames (CMD+W closes the window)
        Visualizer.stream[process_frame](video_capture, "Video")

    except error:
        print(error)
