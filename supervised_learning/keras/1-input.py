#!/usr/bin/env python3
""" build a neural network with keras library"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of the inpute features to the network
    layer is the number of nodes in each layer of the netwrok
    activation is a list containing the activation functionused for each layer of the network
    lambtha is th eL2 regularization parameter
    keep_prob is the propability that a node will kept for the dropout
    Returns: the keras model
    """

    """define the inpute layer"""
    input  = K.layers.Input(shape=(nx,))
    """
    connect the layers then creat a hiden layer as a dense
    that recive the inpute only form the inpute layer
    """
    regulaizer = K.regularizers.l2(lambtha)
    for i in range(len(layers)):
        if i == 0:
            """this conditon check if the layer is the first layer or not"""
            """if it's the first layer then the inpute layer is connected to it"""
            hiden_layer = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=regulaizer)(input)
        if i != len(layers) - 1:
            """this conditon check if the layer is the last layer or not"""
            """if it's not then a dropout layer is added after the curent layer"""
            hiden_layer = K.layers.Dense(layers[i], activation=activations[i], kernel_regularizer=regulaizer)(hiden_layer)
            hiden_layer = K.layers.Dropout(1 - keep_prob)(hiden_layer)
            """1 - keep_prob is the propability that a node wil be dropped out """
            
    output = hiden_layer
    model = K.Model(inputs=input, outputs=output)
    return model