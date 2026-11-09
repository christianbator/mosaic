#
# strided_range.mojo
# mosaic
#
# Created by Christian Bator on 03/15/2025
#

"""Strided ranges of indices."""

from std.os import abort
from std.math import ceildiv


#
# StridedRange
#
struct StridedRange(ImplicitlyCopyable, Movable, Writable):
    """The indices from `start` up to but excluding `end`, `step` apart."""

    #
    # Fields
    #
    var start: Int
    """The first index."""
    var end: Int
    """The end of the range, excluded."""
    var step: Int
    """The distance between consecutive indices."""

    @always_inline
    def count(self) -> Int:
        """The number of indices in the range."""
        return ceildiv(self.end - self.start, self.step)

    #
    # Initialization
    #
    @always_inline
    def __init__(out self, end: Int):
        """Creates the range `[0, end)`."""
        if not (end >= 0):
            abort("Failed StridedRange init requirement: 0 <= end")

        self.start = 0
        self.end = end
        self.step = 1

    @always_inline
    def __init__(out self, start: Int, end: Int):
        """Creates the range `[start, end)`."""
        if not (0 <= start <= end):
            abort("Failed StridedRange init requirement: 0 <= start <= end")

        self.start = start
        self.end = end
        self.step = 1

    @always_inline
    @implicit
    def __init__(out self, tuple: Tuple[Int, Int]):
        """Creates the range `[start, end)` from a `(start, end)` tuple."""
        self = Self(start=tuple[0], end=tuple[1])

    @always_inline
    def __init__(out self, start: Int, end: Int, step: Int):
        """Creates the range `[start, end)` with the given step."""
        if not (0 <= start <= end and step > 0):
            abort("Failed StridedRange init requirement: 0 <= start <= end and step > 0")

        self.start = start
        self.end = end
        self.step = step

    @always_inline
    @implicit
    def __init__(out self, tuple: Tuple[Int, Int, Int]):
        """Creates a range from a `(start, end, step)` tuple."""
        self = Self(start=tuple[0], end=tuple[1], step=tuple[2])

    @always_inline
    def __init__(
        out self,
        slice: Slice,
        default_start: Int,
        default_end: Int,
        default_step: Int,
    ):
        """Creates a range from a slice, using the defaults for any missing start, end or step."""
        self = Self(
            start=slice.start.value() if slice.start else default_start,
            end=slice.end.value() if slice.end else default_end,
            step=slice.step.value() if slice.step else default_step,
        )

    #
    # Writable
    #
    def write_to[W: Writer](self, mut writer: W):
        """Writes the range."""
        writer.write("[StridedRange: (", self.start, ", ", self.end, ", ", self.step, ")]")
