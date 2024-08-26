#!/usr/bin/env python3

"""
Function that finds the best number of clusters for a GMM
using the Bayesian Information Criterion (BIC).
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization

def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Function that finds the best number of clusters for a GMM
    using the Bayesian Information Criterion (BIC).
    """
    if (
        not isinstance(X, np.ndarray) or X.ndim != 2
        or not isinstance(kmin, int) or kmin <= 0
        or kmax is not None and (not isinstance(kmax, int) or kmax <= kmin)
        or not isinstance(iterations, int) or iterations <= 0
        or isinstance(kmax, int) and kmax <= kmin
        or not isinstance(tol, float) or tol < 0
        or not isinstance(verbose, bool)
    ):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        """Undefined, set to maximum possible """
        kmax = n
    if not isinstance(kmax, int) or kmax < 1 or kmax < kmin or kmax > n:
        return None, None, None, None

    b = []
    likelihoods = []
    best_bic = None
    best_results = None
    best_k = None

    """ With each cluster size from kmin to kmax""" 
    for k in range(kmin, kmax + 1):
        """ Find the best fit with the GMM and current cluster size k"""
        
        # Run expectation maximization
        pi, m, S, g, li = expectation_maximization(X, k, iterations, tol, verbose)
        
        # Ensure no covariance matrix is singular by adding regularization if necessary
        for cluster in range(k):
            if np.linalg.det(S[cluster]) == 0:
                S[cluster] += np.eye(S[cluster].shape[0]) * 1e-6  # Add regularization directly

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
        if best_bic is None or bic < best_bic:
            """ Update the return values"""
            best_bic = bic
            best_results = (pi, m, S)
            best_k = k

    likelihoods = np.array(likelihoods)
    b = np.array(b)
    return best_k, best_results, likelihoods, b
