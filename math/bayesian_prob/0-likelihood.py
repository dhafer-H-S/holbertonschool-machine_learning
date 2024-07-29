#!/usr/bin/env python3

"""
This module contains a function to calculate the likelihood of obtaining
data given various hypothetical probabilities of developing
severe side effects
"""

import numpy as np


def likelihood(x, n, P):
    """
    calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects
    x : is the number of patients that develop severe side effects
    n : is the totla number of patients observed
    p : is a 1D numpy.array containing the various hypothetical
    probabilities of developing sever side effects
    """

    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for p in P:
        if p < 0 or p > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    res = 1
    for i in range(x):
        res = res * (n - i) // (i + 1)
    likelihoods = np.zeros_like(P)
    for i, p in enumerate(P):
        likelihoods[i] = res * (p ** x) * ((1 - p) ** (n - x))
    return likelihoods
