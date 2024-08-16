#!/usr/bin/env python3

"""
Function that performs k-means clustering on a dataset.
"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Parameters:
    - X: numpy.ndarray of shape (n, d), the dataset.
    - k: a positive integer, number of clusters.
    - iterations: a positive integer, maximum number of iterations.

    Returns:
    - centroids: numpy.ndarray of shape (k, d), final cluster centroids.
    - clss: numpy.ndarray of shape (n,), index of the cluster each data point belongs to.
    """
    """Step 1: Validate inputs"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    """Initialize centroids with a uniform distribution within the data range"""
    X_min = np.amin(X, axis=0)
    X_max = np.amax(X, axis=0)
    centroids = np.random.uniform(X_min, X_max, size=(k, d))
    if centroids is None:
        return None, None

    """Start iteration process"""
    for _ in range(iterations):
        new_centroids = np.copy(centroids)
        distances = np.linalg.norm(X[:, np.newaxis] - new_centroids, axis=-1)
        clss = np.argmin(distances, axis=-1)
        for j in range(k):
            if X[clss == j].size == 0:
                c[j] = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), (1, X.shape[1]))
            else:
                clss[j] = np.mean(X[clss == j], axis=0)
        if np.array_equal(clss, new_centroids):
            break
    return centroids, clss
