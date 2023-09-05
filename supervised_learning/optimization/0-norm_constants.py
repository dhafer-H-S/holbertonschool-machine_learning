#!/usr/bin/env python3
"""calculates the normalization (standardization) constants of a matrix"""


import numpy as np


def normalization_constants(X):
    """calculates the normalization (standardization) constants of a matrix"""
    """ calculate the mean"""
    mean = np.mean(X, axis=0)
    """calcumate the standar deviation"""
    std = np.std(X, axis=0)
    return mean, std
