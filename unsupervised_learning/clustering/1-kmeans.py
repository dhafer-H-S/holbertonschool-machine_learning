#!/usr/bin/env python3
"""
The k-means init values
to start the algorithm
"""
import numpy as np


def initialize(X, k):
    """
    Using the multivariate uniform
    distribution to have the intial values
    """
    try:
        if k <= 0 or not isinstance(k, int):
            return None
        x_min = np.min(X, axis=0)
        x_max = np.max(X, axis=0)
        init = np.random.uniform(x_min, x_max, size=(k, X.shape[1]))
        return init
    except Exception as e:
        return None


def kmeans(X, k, iterations=1000):
    """
    Using the multivariate uniform
    distribution to have the intial values
    """
    try:
        C = initialize(X, k)
        for _ in range(iterations):
            r = X[:, np.newaxis] - C
            res = np.linalg.norm(r, axis=-1)
            clss = np.argmin(res, axis=-1)
            new_C = np.array([X[clss == i].mean(axis=0) if np.any(
                clss == i) else initialize(X[clss != i], 1)[0] for i in range(k)])
            if np.array_equal(new_C, C):
                return C, clss
            C = new_C
        return C, clss
    except Exception as e:
        return None, None
