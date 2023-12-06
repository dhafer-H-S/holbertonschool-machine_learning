#!/usr/bin/env python3
"""a fucntion that calculates the correlation of a matrix"""
import numpy as np


def correlation(C):
    """function that calculates the correlation of a matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    D = np.sqrt(np.diag(C))
    correlation_matrix = np.zeros(C.shape)

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            correlation_matrix[i][j] = C[i][j] / (D[i] * D[j])

    return correlation_matrix
