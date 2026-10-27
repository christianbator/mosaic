#
# codec.mojo
# mosaic
#
# Created by Christian Bator on 05/03/2025
#

from std.os import abort
from std.ffi import _Global, OwnedDLHandle

from mosaic.utility import dynamic_library_filepath


#
# Backend
#
comptime _libcodec = _Global["libcodec", _load_libcodec]()


def _load_libcodec() -> OwnedDLHandle:
    try:
        return OwnedDLHandle(dynamic_library_filepath("libmosaic-codec"))
    except:
        abort()


#
# ImageFile
#
struct ImageFile(Equatable, ImplicitlyCopyable, Movable, Writable):
    #
    # Supported File Types
    #
    comptime png = Self(ImageFile._png)
    comptime jpeg = Self(ImageFile._jpeg)

    #
    # Fields
    #
    comptime _png = String("png")
    comptime _jpeg = String("jpeg")

    comptime _supported_image_file_types = [Self._png, Self._jpeg]

    var _raw_value: String

    #
    # Initialization
    #
    def __init__(out self, raw_value: String):
        self._raw_value = raw_value

        if raw_value not in materialize[Self._supported_image_file_types]():
            abort("Unsupported image file type: " + raw_value)

    #
    # Properties
    #
    def extension(self) -> String:
        if self == Self.png:
            return ".png"
        elif self == Self.jpeg:
            return ".jpeg"
        else:
            abort("Unimplemented extension() for image file type: " + self._raw_value)

    def valid_extensions(self) -> List[String]:
        if self == Self.png:
            return [".png"]
        elif self == Self.jpeg:
            return [".jpeg", "jpg"]
        else:
            abort("Unimplemented valid_extensions() for image file type: " + self._raw_value)

    #
    # Equatable
    #
    def __eq__(self, other: Self) -> Bool:
        return self._raw_value == other._raw_value

    def __ne__(self, other: Self) -> Bool:
        return not (self == other)

    #
    # Writable
    #
    def write_to[W: Writer](self, mut writer: W):
        writer.write("[ImageFile: ", self._raw_value, "]")
