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
    try:
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            raise ValueError("X must be a 2D numpy array.")
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        n, d = X.shape
        min_val = np.min(X, axis=0)
        max_val = np.max(X, axis=0)
        centroids = np.random.uniform(min_val, max_val, size=(k, X.shape[1]))
        return centroids
    except Exception as e:
        print(f"Error during initialization: {e}")
        return None


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

    """Initialize the centroids using the initialize function"""
    centroids = initialize(X, k)
    if centroids is None:
        return None, None

    n, d = X.shape

    for i in range(iterations):
        """Step 4: Calculate the Euclidean distance between each data point and each centroid"""
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=-1)

        """Step 5: Assign each data point to the closest centroid"""
        clss = np.argmin(distances, axis=-1)

        """Step 6: Calculate new centroids"""
        new_centroids = np.zeros_like(centroids)
        for j in range(k):
            if np.any(clss == j):
                new_centroids[j] = X[clss == j].mean(axis=0)
            else:
                """Reinitialize the centroid if no points are assigned"""
                new_centroids[j] = np.random.uniform(
                    np.min(
                        X, axis=0), np.max(
                        X, axis=0), size=d)

        """Step 7: Check for convergence (if centroids haven't changed, exit early)"""
        if np.all(np.isclose(centroids, new_centroids)):
            break

        centroids = new_centroids

    return centroids, clss
