#!/usr/bin/env python3
"""convert a label vector into a one-hot matrix"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    labels is a one-hot matrix
    classes is the number of classes
    """
    return K.utils.to_categorical(labels, classes)
