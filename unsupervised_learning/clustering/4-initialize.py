#!/usr/bin/env python3

"""
initializes variablees for a gaussian mixture model
"""
import numpy as np
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
        priors = np.full(shape=(k,), fill_value=1 / k)
        mean, _ = kmeans(X, k)
        covariance = np.full(shape=(k, d, d), fill_value=np.identity(d))
        return priors, mean, covariance
    except BaseException:
        return None, None, None
