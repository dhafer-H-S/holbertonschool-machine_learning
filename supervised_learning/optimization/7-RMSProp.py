#!/usr/bin/env python3
""" RMS_prop optimizer that updates the parameters """
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ alpha learning rate """
    """ beta2 the RMSProp weight """
    """ epilson is a small  number to avid divisin by zero """
    """ var containing the variable to be updated """
    """ grad containing the gradient of var """
    """ S is the previous second moment of var """

    """ computes the squared gradient """
    gred_squared = grad ** 2
    """ update the seconde moment (S) using exponential moving average """
    """ s = exp(gred_squared) """
    s_updated = beta2 * s + (1 - beta2) * gred_squared
    """ update the variable using the rms prop updates rule """
    var = var - alpha / (np.sqrt( s + epsilon) * grad)
    return var, s_updated