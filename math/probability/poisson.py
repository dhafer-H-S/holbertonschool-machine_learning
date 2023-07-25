#!/usr/bin/env python3
"""Poisson that represents a poisson distribution of data """


class Poisson:
    """class poission distribution of data"""
    def __init__(self, data=None, lambtha=1.):
        """Poisson that represents a poisson distribution of data """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / float(len(data))

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of “successes”"""
        k = int(k)
        e = 2.7182818285
        if k < 0:
            return 0
        else:
            result = 1.0
            for i in range(1, k+1):
                result = result * i
            return (((self.lambtha**k)*(e**(-self.lambtha))) / result)

    def cdf(self, k):
        """caculates the value of the cdf for a given number of successes"""
        k = int(k)
        if k < 0:
            return 0
        count = 0.0
        for value in range(k + 1):
            count += self.pmf(value)
        return count
