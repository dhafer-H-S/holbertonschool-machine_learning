#!/usr/bin/env python3
""" decay operation using tensorflow predefined function """


import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """
    function to calculate decay operation using
    tensorflow predefined function
    """

    decay = tf.keras.optimizers.schedules.InverseTimeDecay(
        learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)
    return decay
