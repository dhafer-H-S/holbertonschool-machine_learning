#!/usr/bin/env python3

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP

"""
this function calculates the symmetric p affinities for the entire dataset

"""


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """Calculate the p affinities for a given dataset.

    Args:
        X (numpy.ndarray): The input dataset of shape (n, d).
        tol (float, optional): Tolerance value for convergence. Defaults to 1e-5.
        perplexity (float, optional): Perplexity value for determining the number of nearest neighbors. Defaults to 30.0.

    Returns:
        numpy.ndarray: The p affinities matrix of shape (n, n).
    """
    """p affinities"""
    n, d = X.shape
    D, P, betas, H = P_init(X, perplexity)
    for i in range(n):
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        beta_min = None
        beta_max = None
        Hi, Pi = HP(Di, betas[i])
        """Binary search for the correct beta"""
        while np.abs(Hi - H) > tol:
            if Hi > H:
                beta_min = betas[i].copy()
                if beta_max is None:
                    betas[i] *= 2.0
                else:
                    betas[i] = (betas[i] + beta_max) / 2.0
            else:
                beta_max = betas[i].copy()
                if beta_min is None:
                    betas[i] /= 2.0
                else:
                    betas[i] = (betas[i] + beta_min) / 2.0
            Hi, Pi = HP(Di, betas[i])
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = Pi
    P = (P + P.T) / (2 * n)
    return P
