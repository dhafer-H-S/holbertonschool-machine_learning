#!/usr/bin/env python3
"""
function that createsthe training operation for neural network
in tenser flow using gradient descent with momentum
optimization algorithm
"""


import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """
    loss is the loss of the network
    alpha is the learning rate
    beta1 is the momentum weight
    """
    optimizer = tf.compat.v1.train.MomentumOptimizer(alpha, beta1)
    return optimizer
