#
# assertions.mojo
# mosaic
#
# Created by Christian Bator on 05/08/2025
#

from std.os import abort
from std.sys.intrinsics import likely


@always_inline
def _assert[*Ts: Writable](condition: Bool, *messages: *Ts):
    if likely(condition):
        return

    abort(String(*messages))
