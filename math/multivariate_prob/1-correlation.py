#!/usr/bin/env python3
"""a fucntion that calculates the correlation of a matrix"""
import numpy as np


def correlation(C):
    """function that calculates the correlation of a matrix"""
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    D = np.sqrt(np.diag(C))
    correlation_matrix = C / (D * D[:, None])

    return correlation_matrix
