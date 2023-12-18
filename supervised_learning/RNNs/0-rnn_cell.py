#!/usr/bin/env python3
""" RNN class"""

import numpy as np


class RNNCell:
    """a RNN cell class"""

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Wh and bh are for the concatenated hidden state and input data
        Wy and by are for the output
        """
        """class constructor"""
        self.Wh = np.random.randn(h + i, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """function that performs forward propagation for one time step"""
        """Returns: h_next, y"""
        h_next = np.tanh(
            np.matmul(
                np.hstack(
                    (h_prev, x_t)), self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y
        """
        h_next is the next hidden state
        y is the output of the cell
        h_prev is the previous hidden state

        """
