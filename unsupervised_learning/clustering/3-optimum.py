#!/usr/bin/env python3

"""
A function that tests the optimum number of clusters by variance.
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set.
    n is the number of data points
    d is the number of dimensions for each data point
    kmin is a positive integer containing the minimum number
    of clusters to check for (inclusive).
    kmax is a positive integer containing the maximum number
    of clusters to check for (inclusive).
    iterations is a positive integer containing the maximum
    number of iterations for K-means.
    This function should analyze at least 2 different cluster sizes.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < 1):
        return None, None
    if kmax is not None and kmax <= kmin:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    n, d = X.shape
    if kmax is None:
        kmax = n
    results = []
    d_vars = []
    for k in range(kmin, (kmax or kmin) + 1):
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))
        if k == kmin:
            var_min = variance(X, C)
        d_vars.append(var_min - variance(X, C))
    return results, d_vars
