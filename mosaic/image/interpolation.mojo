#
# interpolation.mojo
# mosaic
#
# Created by Christian Bator on 03/20/2025
#

from mosaic.utility import fatal_error


#
# Interpolation
#
@value
struct Interpolation(EqualityComparable, Stringable, Writable):
    alias nearest = Self(0)
    # TODO: Implement these
    # alias bilinear = Self(1)
    # alias bicubic = Self(2)
    # alias lanczos4 = Self(3)
    # alias area = Self(4)

    var _raw_value: Int

    fn __init__(out self, raw_value: Int):
        self._raw_value = raw_value

        if raw_value not in [0]:
            fatal_error("Unsupported interpolation: ", raw_value)

    fn __eq__(self, other: Self) -> Bool:
        return self._raw_value == other._raw_value

    fn __ne__(self, other: Self) -> Bool:
        return not (self == other)

    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[Interpolation: ", self._raw_value, "]")
