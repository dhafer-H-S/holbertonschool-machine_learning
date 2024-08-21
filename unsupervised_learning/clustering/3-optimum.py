#!/usr/bin/env python3

"""
a function that tests the optimum number of clusters by variance
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    kmin is a positive integer containing the minimum number of
    clusters to check for (inclusive)
    kmax is a positive integer containing the maximum number of
    clusters to check for (inclusive)
    iterations is a positive integer containing the maximum number
    of iterations for K-means
    This function should analyze at least 2 different cluster sizes
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0:
        return None, None
    if kmax is None:
        kmax = X.shape
    if kmin > kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None
    results = []
    result_var = []
    for k in range(kmin, kmax + 1):
        centroid, cluster = kmeans(X, k, iterations)
        if k == kmin:
            kmin_var = variance(X, centroid)
        var = variance(X, centroid)
        if var is None:
            return None, None

        d_var = kmin_var - var
        results.append((centroid, cluster))
        result_var.append(d_var)
    return results, result_var
