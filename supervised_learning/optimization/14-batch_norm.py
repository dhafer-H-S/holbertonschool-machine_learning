#!/usr/bin/env python3
"""
a function stabilize and accelerate the training process in
neural network using batch normalization predefined tensorflow function
"""
import tensorflow.compat.v1 as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Create a batch normalization layer for a neural network in TensorFlow.
    Batch Normalization Layer in tensorflow.
    prev: Activated output of the previous layer.
    n: Number of nodes in the layer to be created.
    activation: Activation function to be used on the output of the layer.
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    activation is the activation function that should
    be used on the output of the layer
    Returns: a tensor of the activated output for the layer
    """
    init = tf.keras.initializers.VarianceScaling(mode='fan_avg')
    """
    init is an initializer used to inisialize the weights
    f the dense layer
    """
    layer = tf.keras.layers.Dense(n, kernel_initializer=init)
    """
    desne layer created with 'n' unites and initialized with the
    previously defined initializer
    """
    z = layer(prev)
    gamma = tf.Variable(1., trainable=True)
    beta = tf.Variable(0., trainable=True)
    """ gamma and beta are trainable variables will be used for
    scaling and shifting during batch normalization """
    mean = tf.math.reduce_mean(z, axis=0)
    var = tf.math.reduce_variance(z, axis=0)
    epsilon = 1e-8
    normalized = tf.nn.batch_normalization(
        z, mean, var, beta, gamma, epsilon)
    """
    batch normalization is applied to the pre-activation 'z'
    it uses the calculated mean, variance, beta, gamma and epsilon
    for normalization
    """
    return activation(normalized)
