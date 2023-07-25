#!/usr/bin/env python3
"""calculate the sape of a matrix """


def np_shape(matrix):
    """
    Calculates the shape of a numpy.ndarray.

    Args:
    - matrix: numpy.ndarray, the input array

    Returns:
    - tuple of integers, the shape of the input array
    """
    shape = tuple(map(lambda x: matrix.shape[x], range(matrix.ndim)))
    return shape
