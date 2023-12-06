#!/usr/bin/env python3
"""a class that represent a multivariate normal distribustion"""
import numpy as np

mean_cov = __import__('0-mean_cov').mean_cov


class MultiNormal:
    """multinormal class"""

    def __init__(self, data):
        """function that initialize data"""
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        """
        n is the number f data points
        d is the number of dimmensions in each data point
        """
        if n < 2:
            raise ValueError("data must contain multiple data points")
        self.mean = np.mean(data, axis=1, keepdims=True)
        """cov is of shape (d, d) like if (d, d) = (x, y)"""
        self.cov = ((data - self.mean) @ (data - self.mean).T) / (n - 1)

    def pdf(self, x):
        """functiont that calculates the probability density function"""
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # Calculate the PDF
        x_m = x - self.mean
        pdf = (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(self.cov))) *
               np.exp(-(np.linalg.solve(self.cov, x_m).T.dot(x_m)) / 2))
        return pdf[0, 0]
