#
# stereo.mojo
# mosaic
#
# Created by Christian Bator on 02/20/2025
#

#
# StereoDimensions
#
@value
struct StereoDimensions(Stringable, Writable):

    var width: Int
    var height: Int

    fn __str__(self) -> String:
        return "[StereoDimensions: width = " + String(self.width) + ", height = " + String(self.height) + "]"

    fn write_to[W: Writer](self, mut writer: W):
        writer.write(String(self))
