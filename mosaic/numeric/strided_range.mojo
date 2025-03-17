#
# strided_range.mojo
# mosaic
#
# Created by Christian Bator on 03/15/2025
#

from mosaic.utility import fatal_error


#
# StridedRange
#
@value
@register_passable("trivial")
struct StridedRange(Stringable, Writable):
    #
    # Fields
    #
    var start: Int
    var end: Int
    var step: Int

    #
    # Initialization
    #
    @always_inline
    fn __init__(out self, end: Int):
        self.start = 0
        self.end = end
        self.step = 1

    @always_inline
    fn __init__(out self, start: Int, end: Int):
        self.start = start
        self.end = end
        self.step = 1

    @always_inline
    @implicit
    fn __init__(out self, tuple: Tuple[Int, Int]):
        self.start = tuple[0]
        self.end = tuple[1]
        self.step = 1

    @always_inline
    fn __init__(out self, start: Int, end: Int, step: Int):
        self.start = start
        self.end = end
        self.step = step

    @always_inline
    @implicit
    fn __init__(out self, tuple: Tuple[Int, Int, Int]):
        self.start = tuple[0]
        self.end = tuple[1]
        self.step = tuple[2]

    @always_inline
    fn __init__(
        out self,
        slice: Slice,
        default_start: Int,
        default_end: Int,
        default_step: Int,
    ):
        self.start = slice.start.value() if slice.start else default_start
        self.end = slice.end.value() if slice.end else default_end
        self.step = slice.step.value() if slice.step else default_step

    #
    # Normalization
    #
    fn normalized_in_positive_range(self, end_of_range: Int) raises -> Self:
        var start = self.start + end_of_range if self.start < 0 else self.start
        var end = self.end + end_of_range if self.end < 0 else self.end

        if (0 <= start < end <= end_of_range) and (self.step > 0):
            return Self(start, end, self.step)
        else:
            raise Error("Unable to normalize ", self, " in positive range: [0, ", end_of_range, ")")

    # TODO: Remove this when calling raising function at compile-time works
    fn can_normalize_in_positive_range(self, end_of_range: Int) -> Bool:
        try:
            _ = self.normalized_in_positive_range(end_of_range)
            return True
        except error:
            return False

    #
    # Stringable & Writable
    #
    fn __str__(self) -> String:
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[StridedRange: (", self.start, ", ", self.end, ", ", self.step, ")]")
