import numpy as np

def RowCartesian(arr1, arr2):
    shape1 = arr1.shape[0]
    shape2 = arr2.shape[0]

    arr1 = np.repeat(arr1, repeats=shape2, axis=0)
    arr2 = np.tile(arr2, (shape1, 1))

    return arr1, arr2

