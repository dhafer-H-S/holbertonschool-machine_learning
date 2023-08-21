#!/usr/bin/env python3

import numpy as np

def one_hot_encode(Y, classes):
    """
    Convert a numeric label vector into a one-hot matrix.

    Args:
        Y (numpy.ndarray): Numeric class labels with shape (m,)
        classes (int): Maximum number of classes

    Returns:
        numpy.ndarray: One-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int) or classes <= 0:
        return None

    m = len(Y)
    if m == 0:
        return None
    
    one_hot_matrix = np.zeros((classes, m), dtype=int)
    one_hot_matrix[Y, np.arange(m)] = 1

    """Print the encoded matrix with point separators"""
    for row in one_hot_matrix:
        formatted_row = " . ".join(map(str, row))
        one_hot_matrix = formatted_row
    return one_hot_matrix
