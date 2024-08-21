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
    kmin is a positive integer containing the minimum number of clusters to check for (inclusive).
    kmax is a positive integer containing the maximum number of clusters to check for (inclusive).
    iterations is a positive integer containing the maximum number of iterations for K-means.
    This function should analyze at least 2 different cluster sizes.
    """
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None
        if not isinstance(kmin, int) or kmin <= 0:
            return None, None
        if kmax is not None and (not isinstance(kmax, int) or kmax <= 0):
            return None, None
        if kmax is None:
            kmax = X.shape[0]
        if kmax <= kmin:
            return None, None
        if not isinstance(iterations, int) or iterations <= 0:
            return None, None

        results = []
        result_var = []

        for k in range(kmin, kmax + 1):
            centroid, cluster = kmeans(X, k, iterations)
            if centroid is None or cluster is None:
                return None, None

            var = variance(X, centroid)
            if var is None:
                return None, None

            if k == kmin:
                kmin_var = var

            d_var = var - kmin_var
            results.append((centroid, cluster))
            result_var.append(d_var)

        return results, result_var

    except Exception as e:
        print(f"An error occurred: {e}")
        return None, None
