#
# __init__.mojo
# mosaic
#
# Created by Christian Bator on 02/27/2025
#

"""Platform helpers shared across the library."""

from .system_utilities import optimal_simd_width, unroll_factor, dynamic_library_filepath
from .assertions import _assert
