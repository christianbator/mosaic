#
# device_image.mojo
# mosaic
#
# Created by Christian Bator on 11/08/2026
#

"""Images on the GPU."""

from std.math import ceildiv, isfinite, exp
from std.bit import next_power_of_two
from std.memory import unsafe_memcpy
from std.memory.alloc import unsafe_alloc
from std.sys.info import size_of
from std.ffi import external_call
from std.os import abort
from max.gpu import global_idx
from max.gpu.host import DeviceContext, DeviceBuffer
from layout import ComptimeInt, Idx, TileTensor, row_major, RowMajorLayout

from mosaic.numeric import Size
from mosaic.utility import _assert

from .image import Image
from .border import Border
from .color_space import ColorSpace


#
# DeviceImage
#
struct DeviceImage[color_space: ColorSpace, dtype: DType]:
    """An image on the GPU, in host memory the GPU uses directly on unified memory systems."""

    #
    # Fields
    #
    comptime _Layout = RowMajorLayout[Int, Int, ComptimeInt[Self.color_space.channels()]]
    comptime _Tensor = TileTensor[Self.dtype, Self._Layout, MutUntrackedOrigin]
    comptime _HostImage = Image[Self.color_space, Self.dtype]
    comptime _tile_size = Size(width=16, height=16)
    comptime _bilateral_pixels_per_thread = 4

    var _context: DeviceContext
    # Host memory this image owns and the device uses in place, or None when the image lives in device memory
    var _host_storage: Optional[Pointer[Scalar[Self.dtype], MutUntrackedOrigin]]
    var _buffer: DeviceBuffer[Self.dtype]
    var _tensor: Self._Tensor

    #
    # Initialization
    #
    def __init__(out self, height: Int, width: Int, ctx: DeviceContext) raises:
        """Creates an uninitialized image on the GPU of `ctx`."""
        # Where the device and host share one memory pool, keep the image in host memory so moving data in and out is a
        # CPU copy instead of a slow device transfer
        self = Self(height, width, ctx, _host_memory=ctx.is_host_unified())

    def __init__(out self, height: Int, width: Int, ctx: DeviceContext, *, _host_memory: Bool) raises:
        """Creates an uninitialized image, in host memory when `_host_memory` is set and device memory otherwise."""
        _assert(height > 0 and width > 0, "Height and width must be greater than 0")

        var count = height * width * Self.color_space.channels()
        self._context = ctx

        if _host_memory:
            # The device can only use whole pages of host memory, starting at a page boundary
            var page_size = Int(external_call["getpagesize", Int32]())
            var byte_count = ceildiv(count * size_of[Scalar[Self.dtype]](), page_size) * page_size
            var allocated_count = byte_count // size_of[Scalar[Self.dtype]]()
            var storage = unsafe_alloc[Scalar[Self.dtype]](allocated_count, alignment=page_size)

            try:
                self._buffer = self._context.wrap_host_memory(storage, allocated_count)
            except e:
                storage.unsafe_free()
                raise e^

            self._host_storage = storage
        else:
            self._host_storage = None
            self._buffer = self._context.enqueue_create_buffer[Self.dtype](count)

        # The device address of the memory, which differs from the host address for host memory
        self._tensor = Self._Tensor(
            self._buffer.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin](), Self._layout(height, width)
        )

    def __init__(out self, image: Self._HostImage, ctx: DeviceContext) raises:
        """Creates a copy of `image` on the GPU of `ctx`."""
        self = Self(image.height(), image.width(), ctx)

        # Copy image content
        if self._host_storage:
            # No kernel has used the new memory yet, so the CPU can write it directly
            unsafe_memcpy(dest=self._host_storage.value(), src=image.unsafe_data_ptr(), count=image.samples())
        else:
            self._buffer.enqueue_copy_from(image.unsafe_data_ptr())
            self.synchronize()

    def __deinit__(deinit self):
        """Frees the image's memory, first waiting for pending GPU work when it's host memory."""
        if self._host_storage:
            var storage = self._host_storage.value()

            # Release the device's access to the memory, then wait for that and for every kernel or copy still using it
            # before freeing, since unlike a device buffer, a host free isn't ordered with the device's queued work
            _ = self._buffer^

            try:
                self._context.synchronize()
            except e:
                abort(String("DeviceImage: synchronize failed before freeing host memory: ", e))

            storage.unsafe_free()

    @staticmethod
    def _layout(height: Int, width: Int) -> Self._Layout:
        return row_major(height, width, Idx[Self.color_space.channels()])

    #
    # Properties
    #
    @always_inline
    def height(self) -> Int:
        """The height in pixels."""
        return Int(self._tensor.dim[0]())

    @always_inline
    def width(self) -> Int:
        """The width in pixels."""
        return Int(self._tensor.dim[1]())

    def channels(self) -> Int:
        """The number of channels."""
        return Self.color_space.channels()

    #
    # Copying
    #
    def copy_into(self, mut other: Self) raises:
        """Enqueues a copy of the pixels into `other`, which must have the same size."""
        self._buffer.enqueue_copy_to(other._buffer)

    #
    # Lifetime Management
    #
    def keep(self):
        """Keeps the image alive until this call, for work that uses its memory without owning it."""
        pass

    #
    # Context Synchronization
    #
    @always_inline
    def synchronize(self) raises:
        """Waits for all work enqueued on the image's device context."""
        self._context.synchronize()

    def to_host(self) raises -> Self._HostImage:
        """Waits for pending work and copies the image to the host."""
        var result = Self._HostImage(self.height(), self.width())

        if self._host_storage:
            # Wait for the kernels writing this image, then read its memory directly
            self.synchronize()
            unsafe_memcpy(dest=result.unsafe_data_ptr(), src=self._host_storage.value(), count=result.samples())
        else:
            self._buffer.enqueue_copy_to(result.unsafe_data_ptr())
            self.synchronize()

        return result^

    #
    # Numeric Methods
    #
    def invert(mut self) raises where Self.dtype == DType.uint8:
        """Enqueues an inversion of every sample, replacing `value` with `255 - value`."""
        self._enqueue_invert(result=self._tensor)

    def inverted(self) raises -> Self where Self.dtype == DType.uint8:
        """Returns a copy with every sample inverted, enqueued on the GPU."""
        var result = Self(self.height(), self.width(), self._context)
        self._enqueue_invert(result=result._tensor)

        return result^

    def _enqueue_invert(self, result: Self._Tensor) raises:
        self._context.enqueue_function[Self._invert](
            self._tensor,
            Scalar[Self.dtype].MAX_FINITE,
            result,
            grid_dim=(ceildiv(self.width(), Self._tile_size.width), ceildiv(self.height(), Self._tile_size.height)),
            block_dim=(Self._tile_size.width, Self._tile_size.height),
        )

    @staticmethod
    def _invert(source: Self._Tensor, scalar: Scalar[Self.dtype], result: Self._Tensor):
        var y = global_idx.y
        var x = global_idx.x

        if y >= Int(source.dim[0]()) or x >= Int(source.dim[1]()):
            return

        comptime for channel in range(Self.color_space.channels()):
            result[y, x, channel] = scalar - source[y, x, channel]

    #
    # Common Filters
    #
    def bilateral_filter[
        border: Border
    ](mut self, radius: Int, color_std_dev: Float64, space_std_dev: Float64) raises where Self.dtype == DType.uint8:
        """Smooths while preserving edges, weighting neighbors within `radius` by distance and color difference."""
        self.bilateral_filtered[border](radius, color_std_dev, space_std_dev).copy_into(self)

    def bilateral_filtered[
        border: Border
    ](self, radius: Int, color_std_dev: Float64, space_std_dev: Float64) raises -> Self where Self.dtype == DType.uint8:
        """Returns a copy smoothed while preserving edges, like OpenCV's `bilateralFilter` with `d = 2 * radius + 1`."""
        if radius < 0:
            raise Error("Bilateral filter radius must not be negative")

        # Written as not (> 0) so NaN is rejected too
        if not (color_std_dev > 0) or not (space_std_dev > 0):
            raise Error("Bilateral filter standard deviations must be positive")

        var color_coeff: Float32 = Float32(-0.5 / (color_std_dev**2))
        var space_coeff: Float32 = Float32(-0.5 / (space_std_dev**2))

        # Very small standard deviations overflow the coefficients to -inf, making the weight at distance 0 NaN
        if not isfinite(color_coeff) or not isfinite(space_coeff):
            raise Error("Bilateral filter standard deviations are too small")

        var result = Self(self.height(), self.width(), self._context)

        # Pad on the device first so the filter kernel needs no border logic
        # The right side gets extra columns for the last thread in each row, which filters pixels past the right edge
        # (reading valid padded memory) and only stores the ones inside the image
        comptime pixels_per_thread = Self._bilateral_pixels_per_thread
        # The padded image never leaves the device, so it uses device memory, whose free is ordered after the kernels
        # without waiting for them
        var padded = Self(
            self.height() + 2 * radius,
            self.width() + 2 * radius + pixels_per_thread - 1,
            self._context,
            _host_memory=False,
        )

        self._context.enqueue_function[Self._pad[border]](
            self._tensor,
            Int32(radius),
            padded._tensor,
            grid_dim=(ceildiv(padded.width(), Self._tile_size.width), ceildiv(padded.height(), Self._tile_size.height)),
            block_dim=(Self._tile_size.width, Self._tile_size.height),
        )

        # Window offsets inside the circle of the given radius, as linear sample offsets from the center pixel in the padded
        # image, with a spatial weight for each
        # Sized for the full square window, since the number of points inside the circle isn't known up front
        var side = 2 * radius + 1
        var space_offsets = self._context.enqueue_create_buffer[DType.int32](side**2)
        var space_weights = self._context.enqueue_create_buffer[DType.float32](side**2)
        var space_count = 0

        with space_offsets.map_to_host() as host_offsets, space_weights.map_to_host() as host_space_weights:
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    var distance_squared = dy**2 + dx**2

                    if distance_squared <= radius**2:
                        host_offsets[space_count] = Int32((dy * padded.width() + dx) * Self.color_space.channels())
                        host_space_weights[space_count] = exp(space_coeff * Float32(distance_squared))
                        space_count += 1

        # Color weights for every possible color distance (sum of absolute channel differences, 0 to 255 per channel)
        comptime color_distance_count = Self.color_space.channels() * Int(Scalar[Self.dtype].MAX_FINITE) + 1
        var color_weights = self._context.enqueue_create_buffer[DType.float32](color_distance_count)

        with color_weights.map_to_host() as host_color_weights:
            for color_distance in range(color_distance_count):
                host_color_weights[color_distance] = exp(color_coeff * Float32(color_distance**2))

        self._context.enqueue_function[Self._bilateral_filter](
            padded._tensor,
            Int32(radius),
            space_offsets,
            space_weights,
            Int32(space_count),
            color_weights,
            result._tensor,
            grid_dim=(
                ceildiv(self.width(), Self._tile_size.width * pixels_per_thread),
                ceildiv(self.height(), Self._tile_size.height),
            ),
            block_dim=(Self._tile_size.width, Self._tile_size.height),
        )

        # The kernels only hold an untracked tensor, so keep the padded image alive until both are enqueued
        padded.keep()

        return result^

    @staticmethod
    def _pad[border: Border](source: Self._Tensor, radius: Int32, result: Self._Tensor):
        var y = global_idx.y
        var x = global_idx.x

        if y >= Int(result.dim[0]()) or x >= Int(result.dim[1]()):
            return

        comptime for channel in range(Self.color_space.channels()):
            result[y, x, channel] = Self._bordered_load[border](source, y - Int(radius), x - Int(radius), channel)

    @staticmethod
    def _bilateral_filter(
        padded: Self._Tensor,
        radius: Int32,
        space_offsets: Pointer[Int32, ImmutAnyOrigin],
        space_weights: Pointer[Float32, ImmutAnyOrigin],
        space_count: Int32,
        color_weights: Pointer[Float32, ImmutAnyOrigin],
        result: Self._Tensor,
    ) where Self.dtype == DType.uint8:
        # Each thread filters a run of adjacent pixels in one row, sharing each window point's table loads among them
        comptime pixels = Self._bilateral_pixels_per_thread
        var y = global_idx.y
        var x = global_idx.x * pixels
        var width = Int(result.dim[1]())

        if y >= Int(result.dim[0]()) or x >= width:
            return

        comptime channels = Self.color_space.channels()
        comptime channel_size = next_power_of_two(channels)
        comptime Pixel = SIMD[DType.float32, channel_size]

        # The first center pixel in the padded image, which every neighbor of every pixel is a fixed linear offset from
        var center_ptr = padded.ptr.unsafe_offset(
            ((y + Int(radius)) * Int(padded.dim[1]()) + x + Int(radius)) * channels
        )
        var centers = Array[Pixel, pixels](fill=Pixel(0))

        comptime for pixel in range(pixels):
            comptime for channel in range(channels):
                centers[pixel][channel] = center_ptr[unsafe_offset=pixel * channels + channel].cast[DType.float32]()

        var sums = Array[Pixel, pixels](fill=Pixel(0))
        var weight_sums = Array[Float32, pixels](fill=0)

        for index in range(space_count):
            var offset = Int(space_offsets[unsafe_offset=index])
            var space_weight = space_weights[unsafe_offset=index]

            comptime for pixel in range(pixels):
                var neighbor = Pixel(0)

                comptime for channel in range(channels):
                    neighbor[channel] = center_ptr[unsafe_offset=offset + pixel * channels + channel].cast[
                        DType.float32
                    ]()

                # Float to Int32, since converting to the 64-bit Int is emulated on the GPU and costs as much as the rest of the loop
                var color_distance = abs(neighbor - centers[pixel]).reduce_add().cast[DType.int32]()
                var weight = space_weight * color_weights[unsafe_offset=Int(color_distance)]

                weight_sums[pixel] += weight
                sums[pixel] = neighbor.fma(weight, sums[pixel])

        comptime for pixel in range(pixels):
            if x + pixel < width:
                var filtered_pixel = round(sums[pixel] * (1 / weight_sums[pixel]))

                comptime for channel in range(channels):
                    result[y, x + pixel, channel] = filtered_pixel[channel].cast[Self.dtype]()

    #
    # Filter Utilities
    #
    @staticmethod
    @always_inline
    def _bordered_load[border: Border](source: Self._Tensor, y: Int, x: Int, channel: Int) -> Scalar[Self.dtype]:
        def load(y: Int, x: Int, channel: Int) {imm source} -> Scalar[Self.dtype]:
            return source[y, x, channel]

        return Border.load[border](
            load, y=y, x=x, channel=channel, height=Int(source.dim[0]()), width=Int(source.dim[1]())
        )
