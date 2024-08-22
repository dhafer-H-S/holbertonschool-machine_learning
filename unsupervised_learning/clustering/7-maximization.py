#!/usr/bin/env python3


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
    for i in range(k):
        sub = X - m[i]
        sub_T = sub.T
        S[i] = np.dot((np.sum(g[i]) * sub * sub_T))/ np.sum(g[i])
    return pi, m, S
    