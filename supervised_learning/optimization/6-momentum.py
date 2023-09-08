#!/usr/bin/env python3
"""
function that createsthe training operation for neural network 
in tenser flow using gradient descent with momentum 
optimization algorithm
"""
import tensorflow.compat.v1 as tf

def create_momentum_op(loss, alpha, beta1):
    """
    loss is the loss of the network
    alpha is the learning rate
    beta1 is the momentum weight
    """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)