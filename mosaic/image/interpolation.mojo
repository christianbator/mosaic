#
# interpolation.mojo
# mosaic
#
# Created by Christian Bator on 03/20/2025
#

from os import abort


#
# Interpolation
#
@value
struct Interpolation(EqualityComparable, Stringable, Writable):
    #
    # Supported Interpolations
    #
    alias nearest = Self(0)
    alias bilinear = Self(1)
    alias bicubic = Self(2)
    alias lanczos4 = Self(3)
    alias area = Self(4)

    #
    # Fields
    #
    var _raw_value: Int

    #
    # Initialization
    #
    fn __init__(out self, raw_value: Int):
        self._raw_value = raw_value

        if raw_value not in [0, 1, 2, 3, 4]:
            abort("Unsupported interpolation: " + String(raw_value))

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
        writer.write("[Interpolation: ")

        if self == Interpolation.nearest:
            writer.write("nearest")
        elif self == Interpolation.bilinear:
            writer.write("bilinear")
        elif self == Interpolation.bicubic:
            writer.write("bicubic")
        elif self == Interpolation.lanczos4:
            writer.write("lanczos4")
        elif self == Interpolation.area:
            writer.write("area")
        else:
            abort("Unimplemented write_to() for interpolation with raw value: " + String(self._raw_value))

        writer.write("]")
