#!/usr/bin/env python3
""" a function that makes prediction using a neural network """
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    network is the network model to test
    data is the input data to test the model with
    verbose is a boolean that determines if output should be printed
    during the prediction process
    Returns: the prediction for the data
    """
    return network.predict(data, verbose=verbose)
