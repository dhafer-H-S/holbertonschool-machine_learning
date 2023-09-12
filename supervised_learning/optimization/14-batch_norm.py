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
    dense_layer = tf.layers.Dense(
        units=n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
        use_bias=False)

    dense_output = dense_layer(prev)

    batch_norm_layer = tf.layers.BatchNormalization(
        epsilon=1e-8,
        trainable=True,
        gamma_initializer=tf.ones_initializer(),
        beta_initializer=tf.zeros_initializer())

    bn_output = batch_norm_layer(dense_output)

    activated_output = activation(bn_output)

    return activated_output
