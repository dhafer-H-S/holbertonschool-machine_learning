#!/usr/bin/env python3
"""class normal propability"""


class Normal:
    """class for normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """initialize the normal distribution"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            rs = (sum([(x - self.mean) ** 2 for x in data]) / len(data)) ** 0.5
            self.stddev = rs

    def z_score(self, x):
        """calculate the z score for a normal distribution probability"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """calculate the x value for a normal distribution probability"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """calculate the propability density function"""
        pi = 3.1415926536
        e = 2.7182818285
        return (1 / (self.stddev * (2 * pi) ** 0.5)) *\
            e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)

    def cdf(self, x):
        """calculate the cumultative distribution function"""
        pi = 3.1415926536
        value = (x - self.mean) / ((2 ** 0.5) * self.stddev)
        erf = (((4 / pi) ** 0.5) *
               (value - (value ** 3) / 3 + (value ** 5) / 10 -
                (value ** 7) / 42 + (value ** 9) / 216))
        cdf = (1 + erf) * 0.5
        return cdf
