import numpy as np


def generate(depth_tensor):
    # take the relative depth from the depth matrix
    # as the absolute value

    # get the dimension output from the tensor
    [_, rows, cols, _] = depth_tensor.shape
    depth_matrix = np.reshape(depth_tensor, [rows, cols])
    print('%s %s' %(rows, cols))


    return

