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
    n, d = X.shape

    # Check if X is a valid 2D numpy array
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None

    # Check if m is a valid 1D numpy array of length d
    if not isinstance(m, np.ndarray) or m.shape != (d,):
        return None

    # Check if S is a valid 2D numpy array of shape (d, d)
    if not isinstance(S, np.ndarray) or S.shape != (d, d):
        return None

    # Calculate the difference
    diff = X - m

    # Inverse and determinant of the covariance matrix
    covariance_inv = np.linalg.inv(S)
    covariance_det = np.linalg.det(S)

    # Calculating the exponent component
    exp_component = np.einsum('ij,jk,ik->i', diff, covariance_inv, diff)
    exp_component = -0.5 * exp_component

    # Numerator
    num = np.exp(exp_component)

    # Denominator
    denom = np.sqrt(((2 * np.pi) ** d) * covariance_det)

    # Probability Density Function
    p = num / denom

    # Ensure no value is below 1e-300
    return np.maximum(p, 1e-300)