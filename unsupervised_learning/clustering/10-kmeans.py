#!/usr/bin/env python3

"""
sklearn kmeans
"""
import sklearn.cluster as sk


def kmeans(X, k):
    """performs kmeans on a data set """
    if len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k >= 0:
        return None, None
    KMeans = sk.KMeans(n_clusters = K)
    sk.KMeans.fit(X)
    C = sk.KMeans.cluster_centers
    clss = sk.KMeans.labels
    return C, clss
