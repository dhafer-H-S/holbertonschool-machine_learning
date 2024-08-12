#!/usr/bin/env python3

"""
function that calcualtes the greadient of the low dimensional embedding 'y'
in the sne algorithm this gradient are used to update the position of
the data point in the lof dimensional space during the optimization process
"""

import numpy as np


def grads(Y, P):
    """
    This function computes the gradients of Y (used to update the points
    in the low-dimensional space) and the Q affinities.
    """
    n, ndim = Y.shape
    
    # Import the Q_affinities function
    Q_affinities = __import__('5-Q_affinities').Q_affinities
    
    # Calculate the Q affinities and the numerator of the Q affinities
    Q, num = Q_affinities(Y)
    
    # Compute the gradient
    PQ_diff = P - Q  # Difference between P and Q
    dY = np.zeros_like(Y)
    
    for i in range(n):
        # Calculate the gradient for the i-th point
        dY[i] = np.sum(np.expand_dims(PQ_diff[:, i] * num[:, i], axis=1) * (Y[i] - Y), axis=0)
    
    return dY, Q
