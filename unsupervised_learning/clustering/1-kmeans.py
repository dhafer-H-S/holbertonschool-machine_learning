#!/usr/bin/env python3

"""
Function that performs k-means on a dataset.
"""
import numpy as np

def kmeans(X, k, iterations=1000):
    """
    Performs K-means clustering on a dataset.
    
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset.
        k: positive integer containing the number of clusters.
        iterations: positive integer containing the maximum number of iterations.
    
    Returns:
        - centroids: numpy.ndarray of shape (k, d), final cluster centroids.
        - clss: numpy.ndarray of shape (n,), index of the cluster each data point belongs to.
    """
    def initialize(X, k):
        """Initialize centroids randomly from the dataset."""
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(k, int) or k <= 0 or k > X.shape[0]:
            return None
        n, d = X.shape
        centroids = X[np.random.choice(n, k, replace=False)]
        return centroids

    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    n, d = X.shape

    for i in range(iterations):
        """Calculate the Euclidean distance between each data point and each centroid"""
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1)

        """Assign each data point to the closest centroid"""
        clss = np.argmin(distances, axis=-1)

        """Calculate new centroids"""
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            if np.any(clss == j):
                new_centroids[j] = X[clss == j].mean(axis=0)
            else:
                """Reinitialize the centroid if no points are assigned"""
                new_centroids[j] = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), size=d)

        """Check for convergence (if centroids haven't changed, exit early)"""
        if np.all(np.isclose(centroids, new_centroids)):
            break

        centroids = new_centroids

    return centroids, clss