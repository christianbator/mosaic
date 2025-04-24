#
# video_capture.mojo
# mosaic
#
# Created by Christian Bator on 04/24/2025
#

from mosaic.image import ColorSpace
from mosaic.video import VideoCapture
from mosaic.visualizer import Visualizer


fn main():
    try:
        # Step 1: Attempt to start video capture for the first available camera in the specifed color space.
        #         You can also specify a device by its name (e.g. "MX Brio").
        var video_capture = VideoCapture[ColorSpace.rgb](0)

        # Step 2: Start the camera capture
        video_capture.start()

        # Step 3: Visualize the frames (CMD+W closes the window)
        Visualizer.stream(video_capture, "Video")

    except error:
        print(error)
