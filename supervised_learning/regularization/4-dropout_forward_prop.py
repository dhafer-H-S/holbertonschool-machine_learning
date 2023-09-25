#!/usr/bin/env python3
"""a function that conducts forword propagation using dropout"""
import numpy as np

def dropout_forward_prop(X, weights, L, keep_prob):
    """
    Conduct forward propagation using dropout for a neural network.

    Args:
        X: Input data as a numpy array of shape (nx, m).
           - nx: Number of input features.
           - m: Number of data points.
        weights: Dictionary of weights and biases for the neural network.
        L: Number of layers in the network.
        keep_prob: Probability that a node will be kept during dropout.

    Returns:
        A dictionary containing the outputs of each layer and dropout masks.
    """
    output_layer = {}
    dropouts = {}
    output_layer['A0'] = X
    for i in range(1, L + 1):
        if i < L:
            """ Activation with tanh activation function """
            
            Z = np.dot(weights['W' + str(i)], output_layer['A' + str(i - 1)]) + weights['b' + str(i)]
            A = np.tanh(Z)

            dropout_mask = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob)
            A *= dropout_mask / keep_prob

            output_layer['A' + str(i)] = A
            dropouts['D' + str(i)] = dropout_mask

            
        else:
            """ using softmax activation """
            Z = np.dot(weights['W' + str(i)], X) + weights['b' + str(i)]
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

            output_layer['A' + str(i)] = A

    return output_layer, dropouts
