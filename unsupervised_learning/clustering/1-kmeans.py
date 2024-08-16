#!/usr/bin/env python3
""" 1. K-means """
import numpy as np


def kmeans(X, k, iterations=1000):
    """ performs K-means on a dataset """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    low = np.min(X, axis=0)
    high = np.max(X, axis=0)
    C = np.random.uniform(low, high, (k, d))

    clss = np.zeros(n, dtype=int)

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - C, axis=2)
        new_clss = np.argmin(distances, axis=1)

        if np.array_equal(clss, new_clss):
            break
        clss = new_clss

        new_C = np.array([X[clss == j].mean(axis=0) if np.any(
            clss == j) else np.random.uniform(low, high) for j in range(k)])

        if np.allclose(C, new_C):
            break
        C = new_C

    return C, clss