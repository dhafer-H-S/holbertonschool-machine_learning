#!/usr/bin/env python3
"""Create Dropout layer in tensorflow"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """
    Create a layer of a neural network using dropout.
    prev is a tensor containing the output of the previous layer
    n is the number of nodes the new layer should contain
    activation is the activation function that should be used on the layer
    keep_prob is the probability that a node will be kept
    Return the output of the new layer
    """

    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode='fan_avg')
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer)(prev)

    if training:
        layer = tf.nn.dropout(layer, rate=1 - keep_prob)

    return layer
