#
# video_capture.mojo
# mosaic
#
# Created by Christian Bator on 02/17/2025
#

from mosaic.image import Image, ImagePointer, ColorSpace

#
# VideoCapture
#
trait VideoCapture:

    alias dtype: DType
    alias color_space: ColorSpace

    fn is_next_frame_available(self) -> Bool:
        ...

    fn next_frame(self) -> ImagePointer[Self.dtype, Self.color_space]:
        ...
    
    fn did_read_next_frame(mut self):
        ...
