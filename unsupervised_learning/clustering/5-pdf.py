#!/usr/bin/env python3

"""
initializes variables for a Gaussian mixture model
"""

import numpy as np


def pdf(X, m, S):
    """
    function that calculates the probability density
    function of a Gaussian distribution
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    n, d = X.shape
    if not isinstance(m, np.ndarray) or m.shape != (d,):
        return None
    if not isinstance(S, np.ndarray) or S.shape != (d, d):
        return None
    diff = X - m
    covariance_inv = np.linalg.inv(S)
    covariance_det = np.linalg.det(S)
    exp_component = np.einsum('ij,jk,ik->i', diff, covariance_inv, diff)
    exp_component = -0.5 * exp_component
    num = np.exp(exp_component)
    denom = np.sqrt(((2 * np.pi) ** d) * covariance_det)
    p = num / denom
    return np.maximum(p, 1e-300)
