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
        return String.write(self)

    fn write_to[W: Writer](self, mut writer: W):
        writer.write("[StereoDimensions: width = ", self.width, ", height = ", self.height, "]")
