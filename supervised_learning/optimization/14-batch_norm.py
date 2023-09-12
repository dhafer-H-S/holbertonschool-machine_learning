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
    dense_layer = tf.keras.layers.Dense(
        n, kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'))

    """Apply the Dense layer to the previous layer"""
    dense_output = dense_layer(prev)

    """Create a BatchNormalization layer with trainable parameters gamma and beta"""
    batch_norm_layer = tf.keras.layers.BatchNormalization(epsilon=1e-8)

    """Apply BatchNormalization to the output of the Dense layer"""
    bn_output = batch_norm_layer(dense_output)

    """Apply the activation function to the normalized output"""
    activated_output = activation(bn_output)

    return activated_output
