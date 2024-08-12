#!/usr/bin/env python3


import numpy as np

def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation on the dataset X.

    Parameters:
    - X: numpy.ndarray of shape (n, d), containing the dataset to be transformed by t-SNE.
    - ndims: The new dimensional representation of X.
    - idims: The intermediate dimensional representation of X after PCA.
    - perplexity: The perplexity parameter for t-SNE.
    - iterations: The number of iterations to run the t-SNE algorithm.
    - lr: The learning rate for gradient descent.

    Returns:
    - Y: numpy.ndarray of shape (n, ndim), containing the optimized low-dimensional transformation of X.
    """

    # Step 1: Reduce dimensionality of X using PCA
    pca = __import__('1-pca').pca
    P_affinities = __import__('4-P_affinities').P_affinities
    grads = __import__('6-grads').grads
    cost = __import__('7-cost').cost

    X_reduced = pca(X, idims)
    
    # Step 2: Calculate the pairwise affinities P
    P = P_affinities(X_reduced, perplexity)
    
    # Step 3: Initialize Y and other variables
    n, d = X.shape
    Y = np.random.randn(n, ndims) * 1e-4
    dY = np.zeros_like(Y)
    iY = np.zeros_like(Y)
    gains = np.ones_like(Y)
    
    # Step 4: Early exaggeration phase
    P *= 4.0
    for i in range(1, iterations + 1):
        # Set momentum based on the iteration
        if i < 20:
            momentum = 0.5
        else:
            momentum = 0.8
        
        # Compute the gradient
        dY = grads(Y, P)
        
        # Update Y
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0))
        gains[gains < 0.01] = 0.01  # Prevent too small gains
        
        iY = momentum * iY - lr * (gains * dY)
        Y += iY
        
        # Re-center Y by subtracting the mean
        Y -= np.mean(Y, axis=0)
        
        # Compute the cost and print every 100 iterations
        if i % 100 == 0:
            current_cost = cost(P, Y)
            print(f"Cost at iteration {i}: {current_cost}")
        
        # End early exaggeration after 100 iterations
        if i == 100:
            P /= 4.0
    
    return Y