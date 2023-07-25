#!/usr/bin/env python3
"""function that concatenates twwo matrices with specific axis"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """function that concatenates twwo matrices with specific axis"""
    return np.concatenate((mat1, mat2), axis=axis)
