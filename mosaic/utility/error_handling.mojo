#
# error_handling.mojo
# mosaic
#
# Created by Christian Bator on 05/08/2025
#

from sys import exit
from sys.intrinsics import likely


@always_inline
fn _assert[*Ts: Writable](condition: Bool, *messages: *Ts):
    if likely(condition):
        return

    print(String(messages), flush=True)
    exit(1)


@no_inline
fn fatal_error(error: Error):
    print(error, flush=True)
    exit(1)


@no_inline
fn fatal_error[*Ts: Writable](*messages: *Ts):
    print(String(messages), flush=True)
    exit(1)
