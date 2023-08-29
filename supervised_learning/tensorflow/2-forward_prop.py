#!/usr/bin/env python3

"""forword propagation"""
import tensorflow.compat.v1 as tf
"""Import the create_layer function"""
create_layer = __import__('1-create_layer').create_layer


""" function of the forword propagation"""


def forward_prop(x, layer_sizes=[], activations=[]):
    """ x is the inpute tensor to the neural network"""
    prev_layer = x
    """
    layer size is the list of integers that specifies
    the number of neurons in eache layer
    """
    for i in range(len(layer_sizes)):
        """ n_ nodes represent the layer size is a list of layers """
        n_nodes = layer_sizes[i]
        """ activation is a list of activation functions one for eache layer"""
        activation = activations[i] if i < len(activations) else None
        """ retrives the activation function for the crrent layer"""

        prev_layer = create_layer(prev_layer, n_nodes, activation)
        """
        create a layer based on the current number of nodes
        and activation functions
        effectively coonects the current layer to the previous
        layer in the network
        """
    """
    the output of thr last layer wich is the result
    of the froward prpagation
    """
    prediction = prev_layer

    return prediction
