#
# main.mojo
# mosaic
#
# Created by Christian Bator on 03/14/2025
#

from mosaic.video import VideoCapture
from mosaic.visualizer import Visualizer

fn main():
    try:
        var video_capture = VideoCapture()
        video_capture.start()

        Visualizer.stream(video_capture, "Video")

    except error:
        print(error)
