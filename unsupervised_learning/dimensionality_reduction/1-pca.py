#!/usr/bin/env python3

"""
pca v2
"""

import numpy as np


def pca(X, ndim):
    """Function that performs PCA on a dataset."""
    X_centered = X - np.mean(X, axis=0)
    cov_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    sorted_eigenvalues = eigenvalues[sorted_idx]
    sorted_eigenvectors = eigenvectors[:, sorted_idx]

    ndim = min(ndim, sorted_eigenvectors.shape[1])

    W = sorted_eigenvectors[:, :ndim]

    # Ensure the sign of the eigenvectors aligns with the expected output
    for i in range(ndim):
        if W[0, i] < 0:  # Flip sign if the first element is negative
            W[:, i] = -W[:, i]

    T = np.dot(X_centered, W)
    return T
