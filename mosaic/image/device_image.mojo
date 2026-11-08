#
# device_image.mojo
# mosaic
#
# Created by Christian Bator on 11/08/2026
#

from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import ComptimeInt, Idx, TileTensor, row_major, RowMajorLayout

from mosaic.numeric import Size

from .image import Image
from .color_space import ColorSpace


#
# DeviceImage
#
struct DeviceImage[color_space: ColorSpace, dtype: DType]:
    #
    # Fields
    #
    comptime Layout = RowMajorLayout[Int, Int, ComptimeInt[Self.color_space.channels()]]
    comptime Tensor = TileTensor[Self.dtype, Self.Layout, MutUntrackedOrigin]
    comptime HostImage = Image[Self.color_space, Self.dtype]

    var _context: DeviceContext
    var _buffer: DeviceBuffer[Self.dtype]
    var _tensor: Self.Tensor

    #
    # Initialization
    #
    def __init__(out self, height: Int, width: Int, ctx: DeviceContext) raises:
        self._context = ctx
        self._buffer = self._context.enqueue_create_buffer[Self.dtype](height * width * Self.color_space.channels())
        self._tensor = Self.Tensor(self._buffer.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), Self._layout(height, width))

    def __init__(out self, image: Self.HostImage, ctx: DeviceContext) raises:
        self._context = ctx
        self._buffer = self._context.enqueue_create_buffer[Self.dtype](image.samples())
        self._tensor = Self.Tensor(
            self._buffer.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), Self._layout(image.height(), image.width())
        )

        # Copy image content
        self._buffer.enqueue_copy_from(image.unsafe_data_ptr())
        self.synchronize()

    @staticmethod
    def _layout(height: Int, width: Int) -> Self.Layout:
        return row_major(height, width, Idx[Self.color_space.channels()])

    #
    # Properties
    #
    @always_inline
    def height(self) -> Int:
        return Int(self._tensor.dim[0]())

    @always_inline
    def width(self) -> Int:
        return Int(self._tensor.dim[1]())

    def channels(self) -> Int:
        return Self.color_space.channels()

    #
    # Lifetime Management
    #
    def keep(self):
        pass

    #
    # Context Synchronization
    #
    @always_inline
    def synchronize(self) raises:
        self._context.synchronize()

    def to_host(self) raises -> Self.HostImage:
        var result = Self.HostImage(self.height(), self.width())

        self._buffer.enqueue_copy_to(result.unsafe_data_ptr())
        self.synchronize()

        return result^

    #
    # Numeric Methods
    #
    def invert(mut self) raises where Self.dtype == DType.uint8:
        self._enqueue_invert(result=self._tensor)

    def inverted(self) raises -> Self where Self.dtype == DType.uint8:
        var result = Self(self.height(), self.width(), self._context)
        self._enqueue_invert(result=result._tensor)

        return result^

    def _enqueue_invert(self, result: Self.Tensor) raises:
        comptime tile_size = Size(width=16, height=16)

        self._context.enqueue_function[Self._invert](
            self._tensor,
            Scalar[Self.dtype].MAX_FINITE,
            result,
            grid_dim=(ceildiv(self.width(), tile_size.width), ceildiv(self.height(), tile_size.height)),
            block_dim=(tile_size.width, tile_size.height),
        )

    @staticmethod
    def _invert(source: Self.Tensor, scalar: Scalar[Self.dtype], result: Self.Tensor):
        var y = global_idx.y
        var x = global_idx.x

        if y >= Int(source.dim[0]()) or x >= Int(source.dim[1]()):
            return

        comptime for channel in range(Self.color_space.channels()):
            result[y, x, channel] = scalar - source[y, x, channel]

    #
    # Common Filters
    #
    # def bilateral_filter[border: Border](mut self, radius: Int, color_std_dev: Float64, space_std_dev: Float64) raises:
    #     self.bilateral_filtered[border](radius, color_std_dev, space_std_dev).copy_into(self)

    # def bilateral_filtered[border: Border](self, radius: Int, color_std_dev: Float64, space_std_dev: Float64) raises -> Self:
    #     var result = Self(height=self.height(), width=self.width())
