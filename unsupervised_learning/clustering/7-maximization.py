#usr/bin/env python3

"""
function that calculates maximization step
in the em algorithm for a GMM
"""

import numpy as np


def maximization(X, g):
    """
    X of shape (n, d) containing the data set
    g of shape (k, n) containing the posterior
    probability's for each data point in each cluster 
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
    n, d = X.shape
    k, n = g.shape
    pi = (np.sum(g, axis=1)) / n
    m = np.dot(g, X) / np.sum(g, axis=1)[:, np.newaxis]
    S = np.zeros((k, d, d))
    sub = X - m[1]
    sub_T = sub.T
    sum_g = np.sum(g, axis=1)
    S = (sum_g * sub * sub_T)/ sum_g
    return pi, m, s
    