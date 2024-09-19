#!/usr/bin/env python3

"""
initialize gaussian process
"""
import numpy as np


class GaussianProcess:
    """ a class that represent a noiseless 1D gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        x_init : numpy.ndarray shape(t, 1)represent
        the input already sampled with the block-box function
        y-init : numpy.ndarray shape(t, 1)represent
        the output of the black-box function for each input X_init
        t : number if initial samples
        l : the length parameter for the kernel
        ( the much bigger the value the much the graph is smoother)
        sigma_f : the standard deviation given output*
        of the block-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        X1 :  numpy.ndarray shape(m, 1)
        X2 : numpy.ndarray shape(n, 1)
        The formula for the squared Euclidean
        distance between two points
        """
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + \
            np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqdist)
