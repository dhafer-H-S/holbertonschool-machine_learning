#!/usr/bin/env python3

"""
function that determins the steady state probabilities of a regular
markov chain
"""

import numpy as np

def regular(P):
    """
    P of shape (n, n) P[i, j] is the probability of transitioning
    from state i to state j
    n number of state in the markov chain
    """
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return None
    n = P.shape[0]
    """
    Check if the matrix is stochastic (i.e., all rows sum to 1)
    """
    if not np.allclose(np.sum(P, axis=1), 1):
        return None
    """Check if the matrix is regular"""
    k = np.linalg.matrix_power(P, n)
    if not np.all(k > 0):
        return None
    """
    k is the number of steps the chain can transition from any
    state to any other state with non zero probability
    """
    """if the matrix is regular if there exisit a k for all i and j """
    I = np.eye(n)
    A = P.T - I
    A = np.vstack([A, np.ones(n)])
    b = np.zeros(n+1)
    b[-1] = 1
    try:
        pi = np.linalg.lstsq(A, b, rcond=None)[0]
        return pi.reshape(1, n)
    except np.linalg.LinAlgError:
        return None
