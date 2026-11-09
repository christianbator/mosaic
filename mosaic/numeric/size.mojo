#
# size.mojo
# mosaic
#
# Created by Christian Bator on 03/25/2025
#

"""Two-dimensional sizes."""


@fieldwise_init
struct Size(ImplicitlyCopyable, Movable):
    """A height and width."""

    var height: Int
    """The height."""
    var width: Int
    """The width."""
