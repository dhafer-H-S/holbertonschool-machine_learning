#!/usr/bin/env python3

"""
Function that performs k-means clustering on a dataset.
"""
import numpy as np

def initialize(X, k):
    """
    Initializes cluster centroids for K-means.

    Parameters:
    - X: numpy.ndarray of shape (n, d), the dataset.
    - k: positive integer, the number of clusters.

    Returns:
    - centroids: numpy.ndarray of shape (k, d), containing the initialized centroids.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
        return None
    n, d = X.shape
    centroids = X[np.random.choice(n, k, replace=False)]
    return centroids

def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.

    Parameters:
    - X: numpy.ndarray of shape (n, d), the dataset.
    - k: positive integer, the number of clusters.
    - iterations: positive integer, maximum number of iterations.

    Returns:
    - centroids: numpy.ndarray of shape (k, d), final cluster centroids.
    - clss: numpy.ndarray of shape (n,), index of the cluster each data point belongs to.
    """
    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    for _ in range(iterations):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1)
        clss = np.argmin(distances, axis=-1)
        new_centroids = np.array([X[clss == i].mean(axis=0) if np.any(clss == i) else centroids[i] for i in range(k)])
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    return centroids, clss
