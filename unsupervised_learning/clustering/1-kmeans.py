#!/usr/bin/env python3
"""a function that initializes cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """Returns: a numpy.ndarray of shape (k, d)
    containing the initialized centroids for each cluster,
    or None on failure"""
    try:
        n, d = X.shape
        if not isinstance(k, int) or k <= 0:
            return None
        centroids = np.random.uniform(
            low=np.min(
                X, axis=0), high=np.max(
                X, axis=0), size=(
                    k, d))
        return centroids
    except Exception:
        return None


def kmeans(X, k, iterations=1000):
    """a function that performs K-means on a dataset"""
    try:
        n, d = X.shape
        C = initialize(X, k)
        for i in range(iterations):
            dist = np.linalg.norm(X[:, None] - C, axis=2)
            clss = np.argmin(dist, axis=1)
            new_C = np.zeros((k, d))
            for j in range(k):
                cluster_pts = X[clss == j]
                if cluster_pts.shape[0] > 0:
                    new_C[j] = cluster_pts.mean(axis=0)
                else:
                    new_C[j] = np.random.uniform(
                        np.min(
                            X, axis=0), np.max(
                            X, axis=0), d)
            if np.all(C == new_C):
                break
            C = new_C
        return C, clss
    except BaseException:
        return None, None
