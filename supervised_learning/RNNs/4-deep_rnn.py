#!/usr/bin/env python3
"""deep RNN function tha performs forward propagation for a deep RNN"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    rnn_cells is a list of RNNCell instances of length l that will be used for the forward propagation
    l is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
    t is the maximum number of time steps
    m is the batch size
    i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of shape (l, m, h)
    h is the dimensionality of the hidden state
    """
    t, m, i = X.shape
    l, m, h = h_0.shape
    o = rnn_cells[-1].Wy.shape[1]
    H = np.zeros((t + 1, l, m, h))
    Y = np.zeros((t, m, o))
    H[0] = h_0

    for step in range(t):
        H[step + 1, 0], Y[step] = rnn_cells[0].forward(H[step, 0], X[step])
        for layer in range(1, l):
            H[step +
              1, layer], _ = rnn_cells[layer].forward(H[step, layer], H[step +
                                                                        1, layer -
                                                                        1])

    return H, Y
