#!/usr/bin/env python3
"""
a function def kmeans(X, k, iterations=1000): that performs K-means
on a dataset
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """
        Performs K-means clustering on a dataset.

        Parameters:
        - X: numpy.ndarray of shape (n, d), the dataset.
        n is the number of data points
        d is the number of dimensions for each data point
        - k: a positive integer, number of clusters.
        - iterations: a positive integer, maximum number of iterations.

        Returns:
        - C: numpy.ndarray of shape (k, d), final cluster centroids.
        - clss: numpy.ndarray of shape (n,), index of the cluster
        each data point belongs to.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    n, d = X.shape

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    C = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (k, d))

    for _ in range(iterations):
        C_copy = np.copy(C)

        D = np.linalg.norm(X - C[:, np.newaxis], axis=2)

        clss = np.argmin(D, axis=0)

        for j in range(k):
            if len(X[clss == j]) == 0:
                C[j] = np.random.uniform(np.min(X, axis=0),
                                         np.max(X, axis=0), (1, d))
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        if np.array_equal(C, C_copy):
            return C, clss

    D = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    clss = np.argmin(D, axis=0)
    return C, clss
