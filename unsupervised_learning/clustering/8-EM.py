#!/usr/bin/env python3

"""
Performs the expectation maximization for a GMM.
"""

import numpy as np


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization for a GMM.
    
    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the data set
    - k: positive integer containing the number of clusters
    - iterations: positive integer containing the maximum number
      of iterations for the algorithm
    - tol: non-negative float containing tolerance of the
      log likelihood, used to determine early stopping
    - verbose: boolean that determines if information about
      the algorithm should be printed

    Returns:
    - pi, m, S, g, l on success
    - None, None, None, None, None on failure
    """

    # Validate inputs
    if (not isinstance(X, np.ndarray) or len(X.shape) != 2 or
        not isinstance(k, int) or k <= 0 or k >= X.shape[0] or
        not isinstance(iterations, int) or iterations <= 0 or
        not isinstance(tol, float) or tol < 0 or
        not isinstance(verbose, bool)):
        return None, None, None, None, None

    # Import the required functions
    initialize = __import__('4-initialize').initialize
    expectation = __import__('6-expectation').expectation
    maximization = __import__('7-maximization').maximization

    # Step 1: Initialize parameters
    pi, m, S = initialize(X, k)
    if pi is None or m is None or S is None:
        return None, None, None, None, None

    # Step 2: Expectation-Maximization loop
    prev_log_likelihood = 0
    i = 0  # Instead of using a loop counter
    while i < iterations:
        # Expectation step
        g, log_likelihood = expectation(X, pi, m, S)
        if g is None or log_likelihood is None:
            return None, None, None, None, None

        # Maximization step
        pi, m, S = maximization(X, g)
        if pi is None or m is None or S is None:
            return None, None, None, None, None

        # Check for convergence
        if abs(log_likelihood - prev_log_likelihood) <= tol:
            if verbose:
                print(f"Log Likelihood after {i} iterations: {log_likelihood:.5f}")
            break

        prev_log_likelihood = log_likelihood

        if verbose and (i % 10 == 0 or i == iterations - 1):
            print(f"Log Likelihood after {i} iterations: {log_likelihood:.5f}")

        i += 1  # Increment loop counter

    # Step 3: Return the results
    return pi, m, S, g, log_likelihood
