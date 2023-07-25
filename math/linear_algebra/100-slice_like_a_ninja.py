#!/usr/bin/env python3
"""slice a matrics along a specific axes"""

def np_slice(matrix, axes={}):
    """slice a matrics along a specific axes"""
    slices = [slice(None)] * len(matrix.shape)  # create a slice for each axis
    for axis, index in axes.items():
        if 0 <= axis < len(matrix.shape):  # check if axis is a valid index
            slices[axis] = slice(*index)  # update the slice for the specified axis
    # use the tuple of slices to slice the matrix
    result = matrix[slices[0]]
    for s in slices[1:]:
        result = result[s]
    return result
