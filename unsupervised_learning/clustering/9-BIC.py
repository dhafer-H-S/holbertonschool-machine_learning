#!/usr/bin/env python3

"""
function that finds the best number of clusters for a GMM
using the Bayesian information criterion
"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    function that finds the best number of clusters for a GMM
    using the Bayesian information criterion
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is not None and (not isinstance(kmax, int) or kmax < kmin):
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape
    if kmax is None:
        kmax = n

    l = np.zeros(kmax - kmin + 1)
    b = np.zeros(kmax - kmin + 1)
    best_k = None
    best_result = None
    best_bic = np.inf

    for k in range(kmin, kmax + 1):
        pi, m, S, g, lkhd = expectation_maximization(X, k, iterations, tol, verbose)

        # Handle cases where the EM algorithm fails and returns None
        if lkhd is None:
            l[k - kmin] = np.nan
            b[k - kmin] = np.nan
            continue

        p = k * (d + d * (d + 1) / 2) + k - 1
        bic = p * np.log(n) - 2 * lkhd
        l[k - kmin] = lkhd
        b[k - kmin] = bic

        if bic < best_bic:
            best_bic = bic
            best_k = k
            best_result = (pi, m, S)

    return best_k, best_result, l, b
