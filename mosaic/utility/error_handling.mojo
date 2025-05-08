#
# error_handling.mojo
# mosaic
#
# Created by Christian Bator on 05/08/2025
#

from sys import exit


@no_inline
fn fatal_error(error: Error):
    print(error)
    exit(1)


@no_inline
fn fatal_error[*Ts: Writable](*messages: *Ts):
    print(String(messages), flush=True)
    exit(1)
