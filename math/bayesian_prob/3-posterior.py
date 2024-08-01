#!/usr/bin/env python3

"""
Calculates the posterior probability of obtaining this data with
the various hypothetical probabilities.
This module contains a function To calculate the marginal probability
(also known as the marginal likelihood or evidence)
data given various hypothetical probabilities of developing
severe side effects
"""


import numpy as np


def posterior(x, n, P, Pr):
    """
    Calculates the marginal probability of obtaining the data.

    Parameters:
    x (int): Number of patients that develop severe side effects.
    n (int): Total number of patients observed.
    P (numpy.ndarray): 1D array containing the various hypothetical
    probabilities of developing severe side effects.
    Pr (numpy.ndarray): 1D array containing the prior beliefs of P.

    Returns:
    float: The marginal probability of obtaining x and n.
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
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    for p in P:
        if p < 0 or p > 1:
            raise ValueError("All values in P must be in the range [0, 1]")
    for pr in Pr:
        if pr < 0 or pr > 1:
            raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # Calculate the likelihood
    res = 1
    for i in range(x):
        res = res * (n - i) // (i + 1)
    likelihoods = np.zeros_like(P)
    for i, p in enumerate(P):
        likelihoods[i] = res * (p ** x) * ((1 - p) ** (n - x))

    # Calculate the marginal probability
    marginal_prob = np.sum(likelihoods * Pr)
    posterior = (likelihoods * Pr) / marginal_prob

    return posterior
