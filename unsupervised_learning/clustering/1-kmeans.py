#!/usr/bin/env python3
"""clustering"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """performs K-means on a dataset"""
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    np.random.seed(0)
    C = np.random.uniform(
        np.amin(
            X, axis=0), np.amax(
            X, axis=0), (k, X.shape[1]))
    if C is None:
        return None, None

    for _ in range(iterations):
        C_copy = np.copy(C)

        distances = np.linalg.norm(X[:, None] - C_copy, axis=-1)
        clusters = np.argmin(distances, axis=-1)

        for j in range(k):
            if np.any(clusters == j):
                C[j] = np.mean(X[clusters == j], axis=0)
            else:
                C[j] = np.random.uniform(
                    np.amin(
                        X, axis=0), np.amax(
                        X, axis=0), (1, X.shape[1]))

        if np.all(np.abs(C - C_copy) < 1e-6):
            break

    return C, clusters
