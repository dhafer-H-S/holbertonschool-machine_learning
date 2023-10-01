#!/usr/bin/env python3
"""
a save function that save model in a specific filename 
a load function that load a model from a specific filename
"""
import tensorflow.keras as K


def save_model(network, filename):
    """
    network is the model to save
    filename is the path of the file that the model should be saved
    """
    K.saving.save_model(network, filename)

def load_model(filename):
    """
    filename is the path of the file that the model should be loaded
    """
    return K.saving.load_model(filename)
