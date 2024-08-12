#!/usr/bin/env python3

"""
a full t-sne
"""

import numpy as np


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Step 1: Reduce dimensions using PCA
    """
    pca = __import__('1-pca').pca
    P_affinities = __import__('4-P_affinities').P_affinities
    grads = __import__('6-grads').grads
    cost_fn = __import__('7-cost').cost

    """
    Reduce dimensions to idims using PCA
    """
    X_reduced = pca(X, idims)

    """
    Step 2: Calculate the P affinities
    """
    P = P_affinities(X_reduced, perplexity=perplexity)

    """
    Apply early exaggeration for the first 100 iterations
    """
    P *= 4

    """
    Initialize Y randomly
    """
    n, d = X.shape
    Y = np.random.randn(n, ndims)

    """
    Initialize variables for momentum and gradient descent
    """
    Y_momentum = np.zeros_like(Y)
    a_t = 0.5  # Initial momentum factor

    """
    Step 3: Perform gradient descent for the number of iterations
    """
    for i in range(1, iterations + 1):
        """
        Calculate gradients and Q affinities
        """
        dY, Q = grads(Y, P)

        """
        Update the positions in Y using gradient descent
        """
        Y_momentum = a_t * Y_momentum - lr * dY
        Y += Y_momentum

        """
        Re-center Y by subtracting the mean
        """
        Y -= np.mean(Y, axis=0)

        """
        Adjust momentum after 20 iterations
        """
        if i == 20:
            a_t = 0.8

        """
        Print cost every 100 iterations
        """
        if i % 100 == 0:
            C = cost_fn(P, Q)
            print(f"Cost at iteration {i}: {C}")

        """
        Reduce exaggeration factor after 100 iterations
        """
        if i == 100:
            P /= 4

    """
    Return the final low-dimensional embedding Y
    """
    return Y
