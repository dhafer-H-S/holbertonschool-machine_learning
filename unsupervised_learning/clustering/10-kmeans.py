#!/usr/bin/env python3

"""
sklearn kmeans
"""
import sklearn.cluster as sk


def kmeans(X, k):
    """performs kmeans on a data set """
    if len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    kmeans = sk.KMeans(n_clusters=k)
    kmeans.fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_
    return C, clss
