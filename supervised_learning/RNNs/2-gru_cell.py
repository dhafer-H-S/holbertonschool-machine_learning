#!/usr/bin/env python3
""" a gru funciton that represent a gated recurent unit """
import numpy as np


class GRUCell:
    """ GRU cell class"""

    def __init__(self, i, h, o):
        """ class constructor """
        self.Wz = np.random.randn(h + i, h)
        self.Wr = np.random.randn(h + i, h)
        self.Wh = np.random.randn(h + i, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def sigmoid(x):
        """ sigmoid function """
        return 1 / (1 + np.exp(-x))

    def softmax(x):
        """ tanh function """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ forward propagation for one time step"""
        concat = np.concatenate((h_prev, x_t), axis=1)
        z = sigmoid(np.matmul(concat, self.Wz) + self.bz)
        r = sigmoid(np.matmul(concat, self.Wr) + self.br)
        concate2 = np.concatenate((r * h_prev, x_t), axis=1)
        h = np.tanh(np.matmul(concate2, self.Wh) + self.bh)
        h_next = z * h + (1 - z) * h_prev
        y = softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
