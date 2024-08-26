#!/usr/bin/env python3

"""
function that finds the best number of clusters for a GMM
using the bayesian information criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    function that finds the best number of clusters for a GMM
    using the bayesian information criterion
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    n, d = X.shape
    if kmax is None:
        """Undefined, set to maximum possible """
        kmax = n
    if not isinstance(kmax, int) or kmax < 1 or kmax < kmin or kmax > n:
        return None, None, None, None

    b = []
    likelihoods = []

    """ With each cluster size from kmin to kmax""" 
    for k in range(kmin, kmax + 1):
        """ Find the best fit with the GMM and current cluster size k"""
        pi, m, S, g, li = expectation_maximization(
            X, k, iterations, tol, verbose)

        if pi is None or m is None or S is None or g is None:
            return None, None, None, None
        """ NOTE p is the number of parameters, so k * d with the means,"""
        """k * d * (d + 1) with the covariance matrix, and k - 1 with the priors """
        p = (k * d) + (k * d * (d + 1) // 2) + (k - 1)
        bic = p * np.log(n) - 2 * li

        """ Save log likelihood and BIC value with current cluster size"""
        likelihoods.append(li)
        b.append(bic)

        """ Compare current BIC to best observed BIC"""
        if k == kmin or bic < best_bic:
            """ Update the return values"""
            best_bic = bic
            best_results = (pi, m, S)
            best_k = k

    likelihoods = np.array(likelihoods)
    l = likelihoods
    b = np.array(b)
    return best_k, best_results, l, b
