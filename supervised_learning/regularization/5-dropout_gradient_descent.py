#!/usr/bin/env python3
"""gradient descent to updatte the weights of a neural network with dropout regularization"""
import numpy as np

def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Y contain the correct labels for the data
    Y of shape (classes , m)
    weights dictionnary of weights and biases
    cache dictionnary for the output and dropout masks of each layer 
    alpha learining rate
    keep_prob the propability that node will be kept
    L is the number of layers that a node will be kept
    """
    def tanh(Z):
        """activation function using tanh"""
        return ((np.exp(Z) - np.exp(Z)) / (np.exp(Z) + np.exp(Z)))
    def softmax(Z):
        """activation function using softmax"""
        exp = np.exp(Z)
        return exp / exp.sum(axis=0, keepdims=True)
    def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
        m= Y.shape[1]
        """m number of daat point"""
        for i in range(L, 0, -1):
            if i == L:
                Z = ((np.dot(weights['W' + str(i)], weights['A' + str(i)]) + weights['b' + str(i)]))
                dZ = cache['A' + str(i)] - Y
                """calculate the gradient of the last layer using softmax """
        else:
            """Calculate the gradient of hidden layers using tanh activation"""            
            dA = np.dot(weights['W' + str(i + 1)].T, dZ)
            dA *= cache['D' + str(i)] / keep_prob  # Apply dropout
            dZ = dA * (1 - np.tanh(cache['A' + str(i)])**2)

        """Calculate the gradients for weights and biases"""
        dW = (1 / m) * np.dot(dZ, cache['A' + str(i - 1)].T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        """Update weights and biases"""
        weights['W' + str(i)] -= alpha * dW
        weights['b' + str(i)] -= alpha * db
