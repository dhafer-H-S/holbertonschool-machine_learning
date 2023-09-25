#!/usr/bin/env python3
"""gradient descent to updatte the weights of a neural network with
dropout regularization"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    gradient descent to updatte the weights of a neural network with
    dropout regularization
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for layer in range(L, 0, -1):

        dW = np.dot(dZ, cache['A' + str(layer - 1)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dZ = np.matmul(weights['W' + str(layer)].T, dZ)

        A = cache["A" + str(layer - 1)]

        if layer > 1:
            dZ = dZ *\
                (1 - np.power(A, 2)) * \
                (cache['D' + str(layer - 1)] / keep_prob)

        weights["W" + str(layer)] = weights["W" + str(layer)] - alpha * dW
        weights["b" + str(layer)] = weights["b" + str(layer)] - alpha * db
