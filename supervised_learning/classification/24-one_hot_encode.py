#!/usr/bin/env python3

import numpy as np
""" a function to Convert a numeric label vector into a one-hot matrix """


def one_hot_encode(Y, classes):
    """ check if the Y (numpy.ndarray): Numeric class labels with shape (m,)
        classes (int): Maximum number of classes"""
    if not isinstance(Y, np.ndarray) or len(Y.shape) != 1:
        return None
    if not isinstance(classes, int) or classes < 2 or classes < Y.max():
        return None
    m = Y.shape[0]
    m = len(Y)
    if m == 0:
        return None
    """ numpy.ndarray: One-hot encoding of Y with shape (classes, m),
        or None on failure"""
    one_hot_matrix = np.zeros((classes, m))
    one_hot_matrix[Y, np.arange(m)] = 1
    return one_hot_matrix
