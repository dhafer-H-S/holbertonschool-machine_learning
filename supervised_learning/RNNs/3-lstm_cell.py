#!/usr/bin/env python3
"""long short term memory cell class file"""
import numpy as np


class LSTMCell:
    """long short term memory cell class"""

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        """
        self.Wf = np.random.randn(h + i, h)
        self.Wu = np.random.randn(h + i, h)
        self.Wc = np.random.randn(h + i, h)
        self.Wo = np.random.randn(h + i, h)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def softmax(self, x):
        """softmax activation function"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        x_t is a numpy.ndarray of shape (m, i) that contains
        the data input for the cell
        m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing
        the previous hidden state
        c_prev is a numpy.ndarray of shape (m, h) containing
        the previous cell state
        """
        concat = np.concatenate((h_prev, x_t), axis=1)

        fg = self.sigmoid(np.dot(concat, self.Wf) + self.bf)
        ug = self.sigmoid(np.dot(concat, self.Wu) + self.bu)
        cct = np.tanh(np.dot(concat, self.Wc) + self.bc)
        c_next = fg * c_prev + ug * cct
        ot = self.sigmoid(np.dot(concat, self.Wo) + self.bo)
        h_next = ot * np.tanh(c_next)

        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y

    def sigmoid(self, x):
        """sigmoid activation function"""
        return 1 / (1 + np.exp(-x))
