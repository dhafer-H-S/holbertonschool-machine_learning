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
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    centroids = np.random.uniform(X_min, X_max, size=(k, d))

    """Start iteration process"""
    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)
        new_centroids = np.zeros_like(centroids)

        for j in range(k):
            if np.any(clss == j):
                new_centroids[j] = X[clss == j].mean(axis=0)
            else:
                new_centroids[j] = np.random.uniform(X_min, X_max, size=d)

        """Check for convergence (if centroids haven't changed, exit early)"""
        if np.all(np.isclose(centroids, new_centroids)):
            break

        centroids = new_centroids

    return centroids, clss
