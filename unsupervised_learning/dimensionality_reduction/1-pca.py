#!/usr/bin/env python3

"""
pca v2
"""

import numpy as np


def pca(X, ndim):
    """Function that performs PCA on a dataset."""
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    W = Vt.T[:, :ndim]
    T = np.dot(X_centered, W)
    return T
