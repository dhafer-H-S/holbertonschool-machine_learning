#!/usr/bin/env python3
""" update the gradient descent with momentum optimization algorithms """


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    alpha: is the learning rate
    beta1: is the momentum weight
    var: is a numpy.ndarray containing the variable to be updated
    grad: is a numpy.ndarray containing the gradient of var
    v: is the previous first moment of var
    """
    v = beta1 * v + (1- beta1) * grad
    """ update the momentum v """
    var = var - alpha * v
    return var, v
