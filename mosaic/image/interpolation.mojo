#
# interpolation.mojo
# mosaic
#
# Created by Christian Bator on 03/20/2025
#

from std.os import abort


#
# Interpolation
#
struct Interpolation(Equatable, ImplicitlyCopyable, Movable, Writable):
    #
    # Supported Interpolations
    #
    comptime nearest = Self(0)
    comptime bilinear = Self(1)
    comptime bicubic = Self(2)
    comptime lanczos4 = Self(3)
    comptime area = Self(4)

    #
    # Fields
    #
    var _raw_value: Int

    #
    # Initialization
    #
    def __init__(out self, raw_value: Int):
        self._raw_value = raw_value

        if raw_value not in [0, 1, 2, 3, 4]:
            abort("Unsupported interpolation: " + String(raw_value))

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
