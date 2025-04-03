#
# logging.mojo
# mosaic
#
# Created by Christian Bator on 04/01/2025
#


fn print_list[T: WritableCollectionElement](list: List[T]):
    if len(list) == 0:
        print("[]")
    else:
        var result = String("[")

        for i in range(len(list) - 1):
            result.write(list[i], " ")

        result.write(list[-1])
        result.write("]")

        print(result)
