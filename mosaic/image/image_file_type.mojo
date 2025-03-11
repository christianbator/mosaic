#
# image_file_type.mojo
# mosaic
#
# Created by Christian Bator on 03/11/2025
#

from os import abort

#
# ImageFileType
#
@value
struct ImageFileType(EqualityComparable, Stringable, Writable):

    alias png = Self(ImageFileType._png)
    alias jpeg = Self(ImageFileType._jpeg)

    alias _png = String("png")
    alias _jpeg = String("jpeg")

    alias _supported_image_file_types = [Self._png, Self._jpeg]

    var _raw_value: String

    fn __init__(out self, raw_value: String):
        self._raw_value = raw_value

        if raw_value not in Self._supported_image_file_types:
            abort("Unsupported image file type: ", raw_value)

    fn extension(self) -> String:
        if self == Self.png:
            return ".png"
        elif self == Self.jpeg:
            return ".jpeg"
        else:
            abort("Unimplemented extension() for image file type: ", self._raw_value)
            while True:
                pass

    fn __eq__(self, other: Self) -> Bool:
        return self._raw_value == other._raw_value

    fn __ne__(self, other: Self) -> Bool:
        return not(self == other)
    
    fn __str__(self) -> String:
        return "[ImageFileType: " + self._raw_value + "]"
    
    fn write_to[W: Writer](self, mut writer: W):
        writer.write(String(self))
