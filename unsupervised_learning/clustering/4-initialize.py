#!/usr/bin/env python3

"""
initializes variablees for a gaussian mixture model
"""
import numpy as np
from sklearn.mixture import GaussianMixture
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Initializes variables for a Gaussian Mixture Model

    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the data set
    - k: positive integer containing the number of clusters

    Returns:
    - pi: numpy.ndarray of shape (k,) containing the priors for
    each cluster
    - m: numpy.ndarray of shape (k, d) containing the centroid
    means for each cluster
    - S: numpy.ndarray of shape (k, d, d) containing the covariance
    matrices for each cluster
    - Returns None, None, None on failure
    """
    try:
        n, d = X.shape
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None, None, None
        if not isinstance(k, int) or k <= 0:
            return None, None, None
        pi = np.full(shape=(k,), fill_value=1 / k)
        m, _ = kmeans(X, k)
        S = np.array([np.identity(d) for _ in range(k)])
        return pi, m, S
    except BaseException:
        return None, None, None
