#!/usr/bin/env python3
"""Build a neural network with the Keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Build a neural network with the Keras library.

    nx is the number of input features to the network

    layers is a list containing the number of nodes in
    each layer of the network

    activations is a list containing the activation functions
    used for each layer of the network

    lambtha is the L2 regularization parameter

    keep_prob is the probability that a node will be kept for dropout

    Returns: the keras model
    """
    # Define input layer
    input_layer = K.layers.Input(shape=(nx,))

    # Define regularization
    regularizer = K.regularizers.l2(lambtha)

    # Define hidden layers
    prev_layer = input_layer
    for i in range(len(layers)):
        # Define dense layer
        dense_layer = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=regularizer)(prev_layer)

        # Add dropout layer if not last layer
        if i != len(layers) - 1:
            dropout_layer = K.layers.Dropout(1 - keep_prob)(dense_layer)
            prev_layer = dropout_layer
        else:
            prev_layer = dense_layer

    # Define output layer
    output_layer = prev_layer

    # Create model
    model = K.Model(inputs=input_layer, outputs=output_layer)

    return model
