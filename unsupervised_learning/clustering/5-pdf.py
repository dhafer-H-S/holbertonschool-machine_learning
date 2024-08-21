#!/usr/bin/env python3

"""
initializes variablees for a gaussian mixture model
"""
import numpy as np


def pdf(X, m, S):
    """
    function that calculated the probability density
    function of gaussian distribution
    """
    try:
        n, d = X.shape
        if not isinstance(X, np.ndarray) or len(X.shape) != 2:
            return None
        if not isinstance(m, np.ndarray) or m == None:
            return None
        if not isinstance(S, np.ndarray) or len(S.shape) != 2:
            return None
        diff = X - m
        covariance_inv = np.linalg.inv(S)
        covariance_det = np.linalg.det(S)
        """
        np.einsum is used here for efficient computation.
        It performs the necessary matrix multiplication and
        summation in one step
        """
        exp_component = np.einsum('ij,jk,ik->i', diff, covariance_inv, diff)
        exp_component = -0.5 * exp_component
        num = np.exp(exp_component)
        denom = np.sqrt((2 ** pi) ** d * covariance_det)
        p = num / denom
        return np.maximum(p, 1e-300)
    except:
        return None
