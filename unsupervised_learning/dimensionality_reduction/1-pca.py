#!/usr/bin/env python3

"""
pca v2
"""

import numpy as np


def pca(X, ndim):
    """fuction that performs PCA on a dataset"""
    """normalisation for values"""
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_idx]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]
    W = sorted_eigenvectors[:, :ndim]
    T = np.dot(X_centered, W)
    return T
