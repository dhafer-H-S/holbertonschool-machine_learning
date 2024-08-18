#!/usr/bin/env python3

"""
a function that calculates variance
"""

import numpy as np


def variance(X, C):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    n is the number of data points
    d is the number of dimensions for each data point
    C is a numpy.ndarray of shape (k, d) containing the centroid means for each cluster
    You are not allowed to use any loops
    Returns: var, or None on failure
    var is the total variance
    """
    try:
        X_min_val = np.min(X, axis=0)
        X_max_val = np.max(X, axis=0)
        n, d = X.shape
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(C, np.ndarray) or len(C.shape) != 2:
            return None
        D = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        clss = np.argmin(D, axis=1)
        var = (np.sum((X - C[clss]) ** 2))
        return var
    except Exception:
        return None
