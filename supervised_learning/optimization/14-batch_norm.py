#!/usr/bin/env python3
"""
a function stabilize and accelerate the training process in
neural network using batch normalization predefined tensorflow function
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Create a batch normalization layer for a neural network in TensorFlow.

    prev: Activated output of the previous layer.
    n: Number of nodes in the layer to be created.
    activation: Activation function to be used on the output of the layer.
    """
    """Create a Dense layer with VarianceScaling initializer"""
    dense = tf.layers.Dense(
        units=n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
        use_bias=False)

    mean, variance = tf.nn.moments(dense(prev), axes=0)

    gamma = tf.Variable(tf.ones([n]), dtype=tf.float32)
    beta = tf.Variable(tf.zeros([n]), dtype=tf.float32)

    bn = tf.nn.batch_normalization(
        dense(prev),
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-8)

    return activation(bn)
