#!/usr/bin/env python3
""" shuffle data """


import numpy as np


"""shuffle data points in two matrices the same way"""
def shuffle_data(X, Y):
    """
    x is the data with sahpe (m, nx)
    m is  the number of data point
    nx is th number of features
    """
    """
    y is the data with sahpe (m, nx)
    m is  the number of data point as x
    ny is th number of features
    """
    """
    The permutation() method of the numpy.random module
    returns a re-arranged array while keeping the original array unchanged
    """
    shuffler = np.random.permutation(len(X))
    mat_X_shuffled = X[shuffler]
    mat_Y_shuffled = Y[shuffler]
    return mat_X_shuffled, mat_Y_shuffled