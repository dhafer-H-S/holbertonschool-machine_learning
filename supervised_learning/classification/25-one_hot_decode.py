#!/usr/bin/env python3

import numpy as np


""" a function that converts a one hot matrix into a vector of labels """


def one_hot_decode(one_hot):
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    m, classes = one_hot.shape
    if m == 0 or classes == 0:
        return None

    decoded_labels = np.argmax(one_hot, axis=0)
    return decoded_labels
