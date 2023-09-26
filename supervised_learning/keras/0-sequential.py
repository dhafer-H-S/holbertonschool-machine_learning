#!/usr/bin/env python3
""" build a neural network with keras library"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of input features to the network
    layers is a list containing the number of nodes in each layer of the network
    activation is a list containg the activation function used for each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the propability that node will be kept for dropout
    """
    model = k.Sequential()
    regularizer = k.regulaziers.L2(lambtha)
    for i in range(len(layers)):
        if i == 0
        model.add(k.layers.Dense(layers[i], activation=activations[i], input_shape=(nx,),
                                 kernel_regularizer=regularizer
                                 ))
    else:
        model.add(k.layers.Dense(layers[i], activation=activations[i],  kernel_regularizer=regularizer))
    if i != len(layers) - 1:
        rate = 1 - keep_prob
        model.add(k.layers.Dropout(rate))
    return model