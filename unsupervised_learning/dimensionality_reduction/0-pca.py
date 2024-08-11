#!/usr/bin/env python3


import numpy as np


def pca(X, var=0.95):
    """fuction that performs PCA on a dataset"""
    """
    value : The left singular vectors
    vector : The singular values
    (diagonal elements of the diagonal matrix from SVD)
    sigma is the strength of each value or The right singular vectors
    """
    value, vector, sigma = np.linalg.svd(X)
    """
    cumsum : calculates the cumulative sum of the singular values
    sum calculates the totla sum of all singular values
    the devision gives the cumultative variance
    """
    cumltative_var = np.cumsum(vector) / np.sum(vector)
    r = next((i for i, v in enumerate(cumltative_var) if v >= var)) + 1
    """
    sigma takes the first r columns of the rigt singular vectors which
    correspond to the top 'r' principal components
    """
    W = sigma[:r].T
    return W
