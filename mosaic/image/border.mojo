#
# border.mojo
# mosaic
#
# Created by Christian Bator on 02/27/2025
#

from os import abort


#
# Border
#
@value
struct Border(EqualityComparable, Stringable, Writable):
    #
    # Supported Borders
    #
    alias zero = Self(0)
    alias wrap = Self(1)
    alias reflect = Self(2)

    #
    # Fields
    #
    var _raw_value: Int

    #
    # Initialization
    #
    fn __init__(out self, raw_value: Int):
        self._raw_value = raw_value

        if raw_value not in [0, 1, 2]:
            abort("Unsupported border: " + String(raw_value))

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
