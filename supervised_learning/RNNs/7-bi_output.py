#!/usr/bin/env python3
"""
biderctional RNN function that performs
forward and backward propagation for a bidirectional RNN
"""

import numpy as np


class BidirectionalCell:
    """class that represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """initialization"""
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h + h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """function that performs forward propagation for one time step"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concat, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """function that performsa backward propagtion for one time step"""
        concat = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(concat, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """function that calculates all outputs for the RNN"""
        t, m, h = H.shape
        o = self.by.shape[1]
        Y = np.zeros((t, m, o))
        for step in range(t):
            y = np.matmul(H[step], self.Wy) + self.by
            y = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
            Y[step] = y
        return Y
