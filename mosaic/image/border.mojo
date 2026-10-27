#
# border.mojo
# mosaic
#
# Created by Christian Bator on 02/27/2025
#

from std.os import abort


#
# Border
#
struct Border(Equatable, ImplicitlyCopyable, Movable, Writable):
    #
    # Supported Borders
    #
    comptime zero = Self(0)
    comptime wrap = Self(1)
    comptime reflect = Self(2)

    #
    # Fields
    #
    var _raw_value: Int

    #
    # Initialization
    #
    def __init__(out self, raw_value: Int):
        self._raw_value = raw_value

        if raw_value not in [0, 1, 2]:
            abort("Unsupported border: " + String(raw_value))

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
        writer.write("[Border: ")

        if self == Border.zero:
            writer.write("zero")
        elif self == Border.wrap:
            writer.write("wrap")
        elif self == Border.reflect:
            writer.write("reflect")
        else:
            abort("Unimplemented write_to() for border with raw value: " + String(self._raw_value))

        writer.write("]")
