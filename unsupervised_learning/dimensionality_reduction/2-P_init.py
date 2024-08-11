#!/usr/bin/env python3

"""
Initialize t-SNE
"""

import numpy as np


def P_init(X, perplexity):
    """
    funciton that calculate the p affinities in t sne
    """
    n, d = X.shape
    sum_X = np.sum(np.square(X), axis=1)
    D = np.zeros((n, n))
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return (D, P, betas, H)
