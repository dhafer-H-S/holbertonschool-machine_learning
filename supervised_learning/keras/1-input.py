#!/usr/bin/env python3
""" build a neural network with keras library"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of the inpute features to the network
    layer is the number of nodes in each layer of the netwrok
    activation is a list containing the activation functionused
    for each layer of the network
    lambtha is th eL2 regularization parameter
    keep_prob is the propability that a node will kept for the dropout
    Returns: the keras model
    """
    """define the inpute layer"""
    prev = K.Input(shape=(nx,))
    inputs = prev
    l2 = K.regularizers.L2(lambtha)
    for i, layer in enumerate(layers):
        prev = K.layers.Dense(layer, activation=activations[i],
                              kernel_regularizer=l2)(prev)
        if i != len(layers) - 1:
            prev = K.layers.Dropout(1-keep_prob)(prev)

    model = K.Model(inputs=inputs, outputs=prev)
    return model
