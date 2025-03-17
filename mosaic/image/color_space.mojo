#
# color_space.mojo
# mosaic
#
# Created by Christian Bator on 12/21/2024
#

from mosaic.utility import fatal_error


#
# ColorSpace
#
@value
struct ColorSpace(EqualityComparable, Stringable, Writable):
    #
    # Supported Color Spaces
    #
    alias greyscale = Self(ColorSpace._greyscale)
    alias rgb = Self(ColorSpace._rgb)

    #
    # Fields
    #
    alias _greyscale = String("greyscale")
    alias _rgb = String("rgb")

    alias _supported_color_spaces = [Self._greyscale, Self._rgb]

    var _raw_value: String

    #
    # Initialization
    #
    fn __init__(out self, raw_value: String):
        self._raw_value = raw_value

        if raw_value not in Self._supported_color_spaces:
            fatal_error("Unsupported color space: ", raw_value)

    #
    # Properties
    #
    fn channels(self) -> Int:
        if self == Self.greyscale:
            return 1
        elif self == Self.rgb:
            return 3
        else:
            fatal_error("Unimplemented channels() for color space: ", self._raw_value)
            while True:
                pass

    fn is_display_color_space(self) -> Bool:
        return self in [Self.greyscale, Self.rgb]

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
        writer.write("[ColorSpace: ", self._raw_value, "]")
