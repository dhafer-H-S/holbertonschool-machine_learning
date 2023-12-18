#!/usr/bin/env python3
"""a RNN function that performs forwrod propagation for a simple RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """RNN simple function"""
    t, m, i = X.shape
    h = h_0.shape[1]
    o = rnn_cell.by.shape[1]
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0
    for i in range(t):
        h_next, y = rnn_cell.forward(H[i], X[i])
        H[i + 1] = h_next
        Y[i] = y
    return H, Y
