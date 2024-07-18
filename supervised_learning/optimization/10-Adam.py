#!/usr/bin/env python3
""" adam optimizer using tenserflow predefined function"""


import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """ adam optimizer using tenserflow predefined function"""
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=alpha,
        beta1=beta1,
        beta2=beta2,
        epsilon=epsilon,)
    return optimizer
