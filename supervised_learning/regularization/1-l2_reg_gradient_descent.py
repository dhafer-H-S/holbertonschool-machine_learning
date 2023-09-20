#!/usr/bin/env python3
""" update parameters unsing gradient descent and L2 regularization """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Y : conatins the correct labels for the data witha a shape (classes, M)
    weights : dictionnary containig the weights and biases of neural network
    cache :  a dictionnary containg the output of each layer of the neural
    network
    alpha : learning rate
    lambtha : the L2 regularization
    L : the number of layers og th network
    """
    """
    to update the wieghts and biases using gradient descent we need
    to perform
    a backpropagation to computes the gradient of the cost function with
    respect to the weights and biases
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

    """ number of data points in the neural netwrok """
    m = len(Y[0])
    dz = cache['A'+str(L)] - Y
    for i in range(L, 0, -1):
        dw = (1/m) * np.matmul(dz, cache['A'+str(i-1)].T)\
            + (lambtha / m)*weights['W'+str(i)]
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        da = np.matmul(weights['W'+str(i)].T, dz)
        dz = da * (1-cache['A'+str(i-1)]**2)

        weights['W'+str(i)] = weights['W'+str(i)]\
            - alpha * dw
        weights['b'+str(i)] = weights['b'+str(i)]\
            - alpha * db
        """ retrieve cached values """
        """
        dW is to compute as the product of 1 / m to normalize by the number of
        data points the dot product of dZ and A_prev.T to calculate
        the weight gradient (lambtha / m) * W is to add L2 regularization
        """
        """
        this satnds for the gradient descent for the biases
        """
        """
        update paramaetrs as
        weights
        biases
        add L2 regualrization to the weighs of the current layer"""
