#
# assertions.mojo
# mosaic
#
# Created by Christian Bator on 05/08/2025
#

from os import abort
from sys.intrinsics import likely


@always_inline
fn _assert[*Ts: Writable](condition: Bool, *messages: *Ts):
    if likely(condition):
        return

    abort(String(messages))
