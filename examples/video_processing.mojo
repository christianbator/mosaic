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

        # Step 4: Define a frame processor that's invoked for each new frame, taking a pointer and returning a processed image
        @parameter
        fn process_frame[V: VideoCapturing](image: Pointer[Image[V.color_space, DType.uint8]]) -> Image[ColorSpace.greyscale, DType.uint8]:
            try:
                # Convert to float64 greyscale for processing
                var greyscale = image[].converted_as_type[ColorSpace.greyscale, DType.float64]()

                # Smooth the image for better edge detection
                var blurred = greyscale.gaussian_blurred[Border.reflect](5)

                # Filter and scale output
                var edges = 15 * blurred.filtered[Border.reflect](kernel)

                # Clamp to uint8 range
                edges.clamp(0, 255)
                
                # Return processed image as renderable type
                return edges.as_type[DType.uint8]()

            except error:
                print(error)
                return image[].converted[ColorSpace.greyscale]()

        # Step 5: Visualize the processed frames (CMD+W closes the window)
        Visualizer.stream[process_frame](video_capture, "Video")

    except error:
        print(error)
