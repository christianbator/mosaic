#
# color_space.mojo
# mosaic
#
# Created by Christian Bator on 12/21/2024
#

"""Color spaces."""

from std.os import abort
from std.ffi import c_int


#
# ColorSpace
#
struct ColorSpace(Equatable, ImplicitlyCopyable, Movable, Writable):
    """The color space of an image, which determines its number of channels."""

    #
    # Supported Color Spaces
    #
    comptime greyscale = Self(0)
    """One channel of grey."""
    comptime rgb = Self(1)
    """Three channels: red, green and blue."""

    #
    # Fields
    #
    var _raw_value: Int

    #
    # Initialization
    #
    def __init__(out self, raw_value: Int):
        """Creates a color space from its raw value, aborting for unsupported values."""
        self._raw_value = raw_value

        if raw_value not in [0, 1]:
            abort("Unsupported color space raw value: " + String(raw_value))

    #
    # Properties
    #
    def raw_value(self) -> Int:
        """The raw value."""
        return self._raw_value

    def channels(self) -> Int:
        """The number of channels."""
        if self == Self.greyscale:
            return 1
        elif self == Self.rgb:
            return 3
        else:
            abort("Unimplemented channels() for color space: " + String(self))

    def is_display_color_space(self) -> Bool:
        """Whether the visualizer can show this color space without converting it."""
        return self in [Self.greyscale, Self.rgb]

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
        """Writes the color space's name."""
        writer.write("[ColorSpace: ")

        if self == Self.greyscale:
            writer.write("greyscale")
        elif self == Self.rgb:
            writer.write("rgb")
        else:
            abort("Unimplemented write_to() for color space with raw value: " + String(self._raw_value))

        writer.write("]")
