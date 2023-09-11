#!/usr/bin/env python3
""" decay operation using tensorflow predefined function """


import tensorflow.compat.v1 as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ function to calculate decay operation using tensorflow predefined function """

    decay = tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_step=decay_step,
        decay_rate=decay_rate)
    return decay
