#!/usr/bin/env python3
"""deep RNN function tha performs forward propagation for a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    rnn_cells is a list of RNNCell instances of length l that will be used
    for the forward propagation
    l is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of shape
    (l, m, h)
    h is the dimensionality of the hidden state
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    Y = []
    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0
    for step in range(t):
        for layer in range(l):
            if layer == 0:
                h_next, y = rnn_cells[layer].forward(H[step, layer], X[step])
            else:
                h_next, y = rnn_cells[layer].forward(H[step, layer], h_next)
            H[step + 1, layer] = h_next
        Y.append(y)
    Y = np.array(Y)
    return H, Y
