#!/usr/bin/env python3
""" import numpy modul as np"""
import numpy as np


def one_hot_decode(one_hot):
    """ a function that converts a one hot matrix into a vector of labels """
    """ check methode for the one hot is it a ndarray or not and is it
    a shape more then two or not"""
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    """ m and classes shoul be from a shape of one_hot"""
    m, classes = one_hot.shape
    if m == 0 or classes == 0:
        return None
    """ return an output with a shape of one_hot"""
    decoded_labels = np.argmax(one_hot, axis=0)
    return decoded_labels
