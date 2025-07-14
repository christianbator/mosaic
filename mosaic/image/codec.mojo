#
# codec.mojo
# mosaic
#
# Created by Christian Bator on 05/03/2025
#

from os import abort
from sys.ffi import _Global, _OwnedDLHandle

from mosaic.utility import dynamic_library_filepath


#
# Backend
#
alias _libcodec = _Global["libcodec", _OwnedDLHandle, _load_libcodec]()


fn _load_libcodec() -> _OwnedDLHandle:
    try:
        return _OwnedDLHandle(dynamic_library_filepath("libmosaic-codec"))
    except:
        return abort[_OwnedDLHandle]()


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
            abort("Unsupported image file type: " + raw_value)

    #
    # Properties
    #
    fn extension(self) -> String:
        if self == Self.png:
            return ".png"
        elif self == Self.jpeg:
            return ".jpeg"
        else:
            return abort[String]("Unimplemented extension() for image file type: " + self._raw_value)

    fn valid_extensions(self) -> List[String]:
        if self == Self.png:
            return List[String](".png")
        elif self == Self.jpeg:
            return List[String](".jpeg", "jpg")
        else:
            return abort[List[String]]("Unimplemented valid_extensions() for image file type: " + self._raw_value)

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
