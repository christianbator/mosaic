#
# video_capturing.mojo
# mosaic
#
# Created by Christian Bator on 03/25/2025
#

from memory import Pointer

from mosaic.image import Image, ColorSpace


#
# VideoCapturing
#
trait VideoCapturing:
    alias dtype: DType
    alias color_space: ColorSpace

    fn is_next_frame_available(self) -> Bool:
        ...

    fn next_frame(self) -> Pointer[Image[Self.dtype, Self.color_space], ImmutableAnyOrigin]:
        ...

    fn did_read_next_frame(mut self):
        ...
