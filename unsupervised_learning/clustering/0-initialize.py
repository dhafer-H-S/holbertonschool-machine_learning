#!/usr/bin/env python3

"""
initliaze cluster centred for k means
"""

import numpy as np


def initialize(X, k):
    """
    x: containing the data set that will be used for k-means
    k: is a positive integer containing the number of cluster
    """
    
    if not isinstance(X, np.ndarray)or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None
    n, d = X.shape
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)
    centroids = np.random.uniform(min_val, max_val, (k , d))
    return centroids
