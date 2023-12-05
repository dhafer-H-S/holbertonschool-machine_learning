#!/usr/bin/env python3
"""
mean and covarience : a fucntion that calculates
the covariance of a matrix
"""
import numpy as np


def mean_cov(X):
    """
    X is the numpy.ndarray f shape (n, d) contatning the data set
    n: number of the data points
    d : is the number of dimensions in each data point
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    n, d = X.shape
    mean = np.mean(X, axis=0, keepdims=True)
    X = X - mean
    cov = np.matmul(X.T, X) / (n - 1)
    return mean, cov
