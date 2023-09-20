#!/usr/bin/env python3
""" update parameters unsing gradient descent and L2 regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Y : conatins the correct labels for the data witha a shape (classes, M)
    weights : dictionnary containig the weights and biases of neural network
    cache :  a dictionnary containg the output of each layer of the neural network
    alpha : learning rate
    lambtha : the L2 regularization
    L : the number of layers og th network
    """
    """
    to update the wieghts and biases using gradient descent we need to perform
    a backpropagation to computes the gradient of the cost function with respect
    to the weights and biases
    rhis involves moving backward throught the network from the output layer to
    the inpute layer
    """
    """
    update the weights and biases using L2 regularization
    wee need to substract a term that depends on the current weight and lambtha
    """
    """
    The neural network uses tanh activations on each layer except the last
    , which uses a softmax activation
    """
    m = Y.shape[1]
    grad_cache = {}
    """ number of data points in the neural netwrok """
    for i in range(1, L + 1):
        """ loop through layers"""
        A_prev = cache[f'A{i - 1}']
        A = cache[f'A{i}']
        W = weights[f'W{i}']
        """ retrieve cached values """
        if i == L:
            dZ = A - Y
        else:
            dZ = np.dot(
                weights[f'W{i + 1}'].T, grad_cache[f'dZ{i + 1}']) * (1 - np.power(A, 2))
            """ calculates the gradient of the loss """
        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambtha / m) * W
        """
        dW is to compute as the product of 1 / m to normalize by the number of data points
        the dot product of dZ and A_prev.T to calculate the weight gradient
        (lambtha / m) * W is to add L2 regularization
        """
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        """
        this satnds for the gradient descent for the biases
        """
        weights[f'W{i}'] -= alpha * dW
        weights[f'b{i}'] -= alpha * db
        weights[f'W{i}'] -= alpha * (lambtha / m) * weights[f'W{i}']
        """
        update paramaetrs as
        weights
        biases
        add L2 regualrization to the weighs of the current layer"""
