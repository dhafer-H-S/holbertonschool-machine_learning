#!/usr/bin/env python3
"""a function that conducts forword propagation using dropout"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    X : containg the inpute data for the neural network
    X shape(nx, m)
    nx number of input feature
    m is the number of data points

    weights :  is a dictionary of the weights and biases of the neural network
    L: number of layers
    keep_prob : propability that node will kept
    """
    output_layer = {}
    dropouts = {}
    for i in range(1, L + 1):
        if i < L:
            Z = np.dot(weights['W' + str(i)], X) + weights['b' + str(i)]
            activation = np.tanh(Z)
            """activation with tanh activation function"""
            dropout_mask = (
                np.random.rand(
                    activation.shape[0],
                    activation.shape[1]) < keep_prob)
            """dropout"""
            activation *= dropout_mask / keep_prob
            output_layer['activation' + str(i)] = activation
            dropout_mask['D' + str(i)] = dropout_mask

            X = activation
        else:
            Z = np.dot(weights['W' + str(i)], X) + weights['b' + str(i)]
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)

            output_layer['activation' + str(i)] = A
    return {'output_layer': output_layer, 'dropout_mask': dropout_mask}
