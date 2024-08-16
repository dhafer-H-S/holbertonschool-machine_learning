#!/usr/bin/env python3

import numpy as np


def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Parameters:
    - X: numpy.ndarray of shape (n, d), the dataset.
    - k: a positive integer, number of clusters.
    - iterations: a positive integer, maximum number of iterations.

    Returns:
    - C: numpy.ndarray of shape (k, d), final cluster centroids.
    - clss: numpy.ndarray of shape (n,), index of the cluster each data point belongs to.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    n, d = X.shape

    """Initialize centroids"""
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    centroids = np.random.uniform(min_vals, max_vals, size=(k, d))

    for _ in range(iterations):
        """Compute distances and assign clusters"""
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        clss = np.argmin(distances, axis=1)

        """Update centroids"""
        new_centroids = np.array([X[clss == j].mean(axis=0) if np.any(
            clss == j) else np.random.uniform(min_vals, max_vals, size=d) for j in range(k)])

        """Check for convergence"""
        if np.all(np.isclose(new_centroids, centroids)):
            break

        centroids = new_centroids

    return centroids, clss
