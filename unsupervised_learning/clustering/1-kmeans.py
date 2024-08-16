#!/usr/bin/env python3

"""
Function that performs k-means on a dataset.
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids using a uniform distribution.

    Parameters:
    - X: numpy.ndarray of shape (n, d), the dataset.
    - k: a positive integer, number of clusters.

    Returns:
    - centroids: numpy.ndarray of shape (k, d), containing the initialized centroids.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    n, d = X.shape
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    centroids = np.random.uniform(min_val, max_val, (k, d))
    return centroids


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
    if not isinstance(
            X,
            np.ndarray) or len(
            X.shape) != 2 or not isinstance(
                k,
            int) or k <= 0:
        return None, None

    """nitialize the centroids using the initialize function"""
    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    n, d = X.shape

    for i in range(iterations):
        """Step 4: Calculate the Euclidean distance between each data point and each centroid"""
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        """Step 5: Assign each data point to the closest centroid"""
        clss = np.argmin(distances, axis=1)

        """Step 6: Calculate new centroids"""
        new_centroids = np.array([
            X[clss == j].mean(axis=0) if np.any(clss == j)
            else np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), size=d)
            for j in range(k)
        ])

        """Step 7: Check for convergence (if centroids haven't changed, exit early)"""
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, clss
