#!/usr/bin/env python3
""" standardizes a matrix """


import numpy as np

""" normalize data"""
def normalize(X, m, s):
    """
    x is the data with sahpe (d, nx)
    d is  the number of data point
    nx is th number of features
    """
    """ m is the mean of all features of x"""
    """ s contains the standar deviation of all features of x"""
    """ the function to calculate the normilasied data"""
    normalized_X = (X - m) / s
    return normalized_X
