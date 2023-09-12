#!/usr/bin/env python3
"""
a function stabilize and accelerate the training process in
neural network using batch normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ batch normalize function """
    """
    z : represent the unactivated output of a neural network
    gamma: contain scaling factors used in batch normalization
    gamma ->allows the neural network to learn whether to amplify
            or reduce the normalized values for each feature
    beta : contains offset factor used in batch normalization
    beta -> allows the neural network to learn an offset or
            shift in the data for each feature
    epilson is just used to prevent division by zero when
            calculating the variance during normalization
    """
    """ calculating the mean """
    mean = np.mean(Z, axis=0)
    """ calculating the variance """
    variance = np.var(Z, axis=0)
    """ normalize z """
    Z_normalized = (Z - mean) / np.sqrt(variance + epsilon)
    """ scale and shift the normalized values """
    Z_scaled_shifted = gamma * Z_normalized + beta

    return Z_scaled_shifted
