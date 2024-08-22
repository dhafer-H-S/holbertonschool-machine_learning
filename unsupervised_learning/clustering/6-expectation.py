#!/usr/bin/env python3

"""
function that calculates exception step in the EM
algorithm for a GMM
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    X is a numpy.ndarray of shape (n, d) containing the data set
    pi is a numpy.ndarray of shape (k,) containing the priors for each
    cluster
    m is a numpy.ndarray of shape (k, d) containing the centroid means
    for each cluster
    S is a numpy.ndarray of shape (k, d, d) containing the covariance
    matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    n, d = X.shape
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    k = pi.shape
    if not isinstance(pi, np.ndarray) or pi.shape !=  (len(m),):
        return None, None
    k = pi.shape[0]
    """initialize g that will store the responsabilities"""
    g = np.zeros((k, n))
    """ calculate the posterior probabilities for each cluster"""
    for cluster in range(k):
        g[cluster, :] = pi[cluster] * pdf(X, m[cluster], S[cluster])
    """calcualtes the marginal probablility for each data point"""
    marginal = np.sum(g, axis=0)
    """calculate log likelihood"""
    log_likelihood = np.sum(np.log(marginal))
    posterior = g / marginal
    return posterior, log_likelihood
