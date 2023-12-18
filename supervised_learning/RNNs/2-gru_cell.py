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

    @staticmethod
    def sigmoid(x):
        """ sigmoid function """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """ softmax activation function """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ forward propagation for one time step """
        x = np.concatenate((h_prev, x_t), axis=1)
        z = self.sigmoid(np.matmul(x, self.Wz) + self.bz)
        r = self.sigmoid(np.matmul(x, self.Wr) + self.br)
        x = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(np.matmul(x, self.Wh) + self.bh)
        h_next = (1 - z) * h_prev + z * h_tilde
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y