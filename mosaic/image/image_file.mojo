#
# image_file.mojo
# mosaic
#
# Created by Christian Bator on 03/11/2025
#

from mosaic.utility import fatal_error


#
# ImageFile
#
@value
struct ImageFile(EqualityComparable, Stringable, Writable):
    #
    # Supported File Types
    #
    alias png = Self(ImageFile._png)
    alias jpeg = Self(ImageFile._jpeg)

    #
    # Fields
    #
    alias _png = String("png")
    alias _jpeg = String("jpeg")

    alias _supported_image_file_types = [Self._png, Self._jpeg]

    var _raw_value: String

    #
    # Initialization
    #
    fn __init__(out self, raw_value: String):
        self._raw_value = raw_value

        if raw_value not in Self._supported_image_file_types:
            fatal_error("Unsupported image file type: ", raw_value)

    #
    # Properties
    #
    fn extension(self) -> String:
        if self == Self.png:
            return ".png"
        elif self == Self.jpeg:
            return ".jpeg"
        else:
            fatal_error("Unimplemented extension() for image file type: ", self._raw_value)
            while True:
                pass

    #
    # EqualityComparable
    #
    fn __eq__(self, other: Self) -> Bool:
        return self._raw_value == other._raw_value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    #
    # Stringable & Writable
    #
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[ImageFile: ", self._raw_value, "]")
