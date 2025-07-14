#
# color_space.mojo
# mosaic
#
# Created by Christian Bator on 12/21/2024
#

from os import abort
from sys.ffi import c_int


#
# ColorSpace
#
@value
struct ColorSpace(EqualityComparable, Stringable, Writable):
    #
    # Supported Color Spaces
    #
    alias greyscale = Self(0)
    alias rgb = Self(1)

    #
    # Fields
    #
    var _raw_value: Int

    #
    # Initialization
    #
    fn __init__(out self, raw_value: Int):
        self._raw_value = raw_value

        if raw_value not in [0, 1]:
            abort("Unsupported color space raw value: " + String(raw_value))

    #
    # Properties
    #
    fn raw_value(self) -> Int:
        return self._raw_value

    fn channels(self) -> Int:
        if self == Self.greyscale:
            return 1
        elif self == Self.rgb:
            return 3
        else:
            return abort[Int]("Unimplemented channels() for color space: " + String(self))

    fn is_display_color_space(self) -> Bool:
        return self in [Self.greyscale, Self.rgb]

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
        writer.write("[ColorSpace: ")

        if self == Self.greyscale:
            writer.write("greyscale")
        elif self == Self.rgb:
            writer.write("rgb")
        else:
            abort("Unimplemented write_to() for color space with raw value: " + String(self._raw_value))

        writer.write("]")
