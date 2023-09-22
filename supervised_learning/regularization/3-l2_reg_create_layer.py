#!/usr/bin/env python3
""" a tensorflow layer that includes L2 regularization """
import tensorflow.compat.v1 as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    prev: is a tensor containgn the output of the previous layer
    n: is the number of nodes the new layer should contain
    activation: is the activation function that should be used on the layer
    lambtha: the L2 regularization parameter
    """
    L2_regularization = tf.keras.regularizers.L2(lambtha)
    """a regularier that applies a L2 regularization penalty"""
    layer_weight = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))
    """initializer that adapt tits scale tot the shape og it's inpute tensors"""

    layer = tf.layers.Dense(n, activation=activation,
                            kernel_initializer=layer_weight,
                            kernel_regularizer=L2_regularization)
    return layer(prev)
