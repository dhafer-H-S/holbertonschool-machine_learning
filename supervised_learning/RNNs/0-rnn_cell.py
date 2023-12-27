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
        """performs a forward propagation"""
        h_next = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(h_next, self.Wh) + self.bh)
        y = np.matmul(h_next, self.Wy) + self.by
        """ the use of softmax activation fucntion is to normalize the output"""
        """ the use of exp is to make the output positive"""
        """
        the devision between the exp(y and the sum of exp(y) is to get
        the real value of the output that it gonna be between 0 and 1
        """
        y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        return h_next, y

