#!/usr/bin/env python3

"""
sklearn kmeans
"""
import sklearn.cluster as sk
import numpy as np


def kmeans(X, k):
    """performs kmeans on a data set """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k >= 0:
        return None, None
    sk.KMeans(n_clusters = K)
    sk.KMeans.fit(X)
    C = sk.KMeans.cluster_centers
    clss = sk.KMeans.labels
    return C, clss


    