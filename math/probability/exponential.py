#!/usr/bin/env python3
"""Exponential that represents an exponential distribution"""


class Exponential:
    """Exponential distribution class"""
    def __init__(self, data=None, lambtha=1.):
        """Exponential distribution class initialization for a given data"""

        self.lambtha = float(lambtha)
        if data is None:
            if lambtha <= 0.0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1.0 / (sum(data) / float(len(data)))

    def pdf(self, x):
        """Exponential function to calculate the pdf"""

        e = 2.7182818285
        if x < 0:
            return 0
        return (self.lambtha * e**(-self.lambtha * x))

    def cdf(self, x):
        """calculate the number of cdf of a given time period"""

        e = 2.7182818285
        if x < 0:
            return 0
        return (1 - e**(-self.lambtha * x))
