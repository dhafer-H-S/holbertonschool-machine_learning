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
    layer = tf.layers.dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0, mode=("fan_avg")),
        kernel_reglarization=tf.contrib.layers.l2_regularizer(lambtha)
    )(prev)
    return layer
