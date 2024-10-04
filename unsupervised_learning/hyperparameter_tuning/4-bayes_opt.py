#!/usr/bin/env python3

"""
initialize bayesian optimization
"""

import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    a class bayesian optimization that performs bayesian on
    a noiseless 1 D Gaussian process
    """

    def __init__(
            self,
            f,
            X_init,
            Y_init,
            bounds,
            ac_samples,
            l=1,
            sigma_f=1,
            xsi=0.01,
            minimize=True):
        """
        f is black box function to be optimized
        X_init is numpy ndarray of shape (t, 1) representing the inputs
        already sampled with the black box function
        Y_init  shape (t, 1) represent the output of the black box function
        for each input in X_init
        t is the number of initial samples
        bounds is a tuple of (min,max)representing the bounds of the space
        in which the look for the optimal point
        ac_samples is the number of samples that should be analyzed during
        acquisition
        l is the length parameter for the kernel
        sigma_f is the standard deviation given to the output of the black_box
        function
        xsi is the exploration exploitation factor for a acquisition
        minimize is a bool determining whether optimization should be performed
        for minimization (true) or maximization (false)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """a function that calculates the next best sample location"""
        mu, sigma = self.gp.predict(self.X_s)
        if self.minimize:
            """ determines the optimal sample values based on whether
            the goal is minimize or maximize"""
            mu_sample_opt = np.min(self.gp.Y)
        else:
            mu_sample_opt = np.max(self.gp.Y)
        with np.errstate(divide='warn'):
            """calculate the improvement"""
            imp = mu_sample_opt - mu - self.xsi
            """standardizes the improvement"""
            Z = imp / sigma
            """computes the expected improvement"""
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            """this is to handel the case when
            the standard deviation is zero"""
            EI[sigma == 0.0] = 0.0
        """
        identifies the next best sample point based on
        the maximum expected improvements
        """
        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI
