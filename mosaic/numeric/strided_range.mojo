#
# strided_range.mojo
# mosaic
#
# Created by Christian Bator on 03/15/2025
#

#
# StridedRange
#
@value
@register_passable("trivial")
struct StridedRange:

    var start: Int
    var end: Int
    var step: Int

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
    fn __init__(out self, slice: Slice, default_start: Int, default_end: Int, default_step: Int):
        self.start = slice.start.value() if slice.start else default_start
        self.end = slice.end.value() if slice.end else default_end
        self.step = slice.step.value() if slice.step else default_step
