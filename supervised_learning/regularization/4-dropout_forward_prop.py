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

    for i in range(1, L + 1):
        if i < L:
            Z = np.dot(weights['W' + str(i)], X) + weights['b' + str(i)]
            A = np.tanh(Z)  # Activation with tanh activation function

            dropout_mask = (np.random.rand(A.shape[0], A.shape[1]) < keep_prob)
            A *= dropout_mask / keep_prob

            output_layer['activation' + str(i)] = A
            dropouts['D' + str(i)] = dropout_mask

            X = A
        else:
            Z = np.dot(weights['W' + str(i)], X) + weights['b' + str(i)]
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

            output_layer['activation' + str(i)] = A

    return {'output_layer': output_layer, 'dropout_masks': dropouts}