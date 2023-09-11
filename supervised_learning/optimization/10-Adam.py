#!/usr/bin/env python3
""" adam optimizer using tenserflow predefined function"""


import tensorflow.compat.v1 as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    adam_optimizer = tf.compat.v1.train.AdamOptimizer(
    learning_rate=alpha,
    beta1=beta1,
    beta2=beta2,
    epsilon=epsilon,)
    return adam_optimizer.minimize(loss)

