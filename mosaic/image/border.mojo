#
# border.mojo
# mosaic
#
# Created by Christian Bator on 02/27/2025
#

"""Border modes for reading outside an image."""

from std.os import abort


#
# Border
#
struct Border(Equatable, ImplicitlyCopyable, Movable, Writable):
    """How filters read pixels outside an image's bounds."""

    #
    # Supported Borders
    #
    comptime zero = Self(0)
    """Reads zero outside the image."""
    comptime wrap = Self(1)
    """Wraps around to the opposite edge."""
    comptime reflect = Self(2)
    """Mirrors at the edge without repeating the edge pixel, like OpenCV's `BORDER_REFLECT_101`."""

    #
    # Fields
    #
    var _raw_value: Int

    #
    # Initialization
    #
    def __init__(out self, raw_value: Int):
        """Creates a border from its raw value, aborting for unsupported values."""
        self._raw_value = raw_value

        if raw_value not in [0, 1, 2]:
            abort("Unsupported border: " + String(raw_value))

    #
    # Bordered Loading
    #
    @staticmethod
    @always_inline
    def load[
        dtype: DType, Loader: def(Int, Int, Int) -> Scalar[dtype], //, border: Border
    ](loader: Loader, y: Int, x: Int, channel: Int, height: Int, width: Int) -> Scalar[dtype]:
        """Loads a sample through `loader`, mapping out-of-bounds coordinates according to `border`."""
        comptime if border == Border.zero:
            if y >= 0 and y < height and x >= 0 and x < width:
                return loader(y, x, channel)
            else:
                return Scalar[dtype](0)
        elif border == Border.wrap:
            return loader(Self.wrapped(y, height), Self.wrapped(x, width), channel)
        elif border == Border.reflect:
            return loader(Self.reflected(y, height), Self.reflected(x, width), channel)
        else:
            abort("Unimplemented border type in Border.load()")

    #
    # Index Mapping
    #
    @staticmethod
    @always_inline
    def wrapped(index: Int, size: Int) -> Int:
        """Maps `index` into `[0, size)` by wrapping around."""
        return index % size

    @staticmethod
    @always_inline
    def reflected(index: Int, size: Int) -> Int:
        """Maps `index` into `[0, size)` by reflecting at the edges."""
        # Fast path for indices in bounds
        if UInt(index) < UInt(size):
            return index

        if size == 1:
            return 0

        var period = 2 * (size - 1)
        var folded = abs(index) % period

        return folded if folded < size else period - folded

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
        """Writes the border's name."""
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
