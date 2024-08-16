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
