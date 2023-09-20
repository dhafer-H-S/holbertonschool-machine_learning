#!/usr/bin/env python3
"""
a function that calculates the cost function of a neural network with L2
regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    cost : is the cost of the network without l2 regularization
    lambtha : the regularization parameter
    weights : is a dictionary of the weights and biases
    L : number of layers
    m : number of data point used
    """
    """
    L2 regularization takes the sum of squared residuals
    + the square of the weights * lambtha
    """
    L2_regularization_cost = 0.0
    for i in range(1, L + 1):
        W = weights[f'W{i}']
        """
        retrieve the weights for the current layer from the dictionary weights
        """
        L2_regularization = np.sum(np.square(W))
        """ calculate the L2 regularization for each layer """
        L2_regularization_cost += L2_regularization

    L2_regularization_cost *= (lambtha / (2 * m))
    """
    multiply accumlated regularization cost by the regularization parameter
    """
    total_cost = cost + L2_regularization_cost
    return total_cost
