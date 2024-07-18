#!/usr/bin/env python3
""" adam optimizer using tenserflow predefined function"""


import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """ adam optimizer using tenserflow predefined function"""
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon,)
    return optimizer
