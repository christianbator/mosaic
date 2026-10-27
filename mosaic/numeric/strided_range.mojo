#
# strided_range.mojo
# mosaic
#
# Created by Christian Bator on 03/15/2025
#

from std.os import abort
from std.math import ceildiv


#
# StridedRange
#
struct StridedRange(ImplicitlyCopyable, Movable, Writable):
    #
    # Fields
    #
    var start: Int
    var end: Int
    var step: Int

    @always_inline
    def count(self) -> Int:
        return ceildiv(self.end - self.start, self.step)

    #
    # Initialization
    #
    @always_inline
    def __init__(out self, end: Int):
        if not (end >= 0):
            abort("Failed StridedRange init requirement: 0 <= end")

        self.start = 0
        self.end = end
        self.step = 1

    @always_inline
    def __init__(out self, start: Int, end: Int):
        if not (0 <= start <= end):
            abort("Failed StridedRange init requirement: 0 <= start <= end")

        self.start = start
        self.end = end
        self.step = 1

    @always_inline
    @implicit
    def __init__(out self, tuple: Tuple[Int, Int]):
        self = Self(start=tuple[0], end=tuple[1])

    @always_inline
    def __init__(out self, start: Int, end: Int, step: Int):
        if not (0 <= start <= end and step > 0):
            abort("Failed StridedRange init requirement: 0 <= start <= end and step > 0")

        self.start = start
        self.end = end
        self.step = step

    @always_inline
    @implicit
    def __init__(out self, tuple: Tuple[Int, Int, Int]):
        self = Self(start=tuple[0], end=tuple[1], step=tuple[2])

    @always_inline
    def __init__(
        out self,
        slice: Slice,
        default_start: Int,
        default_end: Int,
        default_step: Int,
    ):
        self = Self(
            start=slice.start.value() if slice.start else default_start,
            end=slice.end.value() if slice.end else default_end,
            step=slice.step.value() if slice.step else default_step,
        )

    #
    # Writable
    #
    def write_to[W: Writer](self, mut writer: W):
        writer.write("[StridedRange: (", self.start, ", ", self.end, ", ", self.step, ")]")
