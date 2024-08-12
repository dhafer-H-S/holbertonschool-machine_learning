#!/usr/bin/env python3

"""
q affinities
"""
import numpy as np


def Q_affinities(Y):
    """
    Y contaning the low dimensional transformation
    """
    n, ndim = Y.shape
    sum_Y = np.sum(np.square(Y), axis=1)
    D = np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y)
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return Q, num
