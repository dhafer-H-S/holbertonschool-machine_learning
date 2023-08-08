#!/usr/bin/env python3

"""deep neural network"""
import numpy as np


class DeepNeuralNetwork:
    """deep neural network"""

    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        layer_size = nx

        for i in range(1, self.L + 1):
            he_et_al = np.sqrt(2 / layer_size)
            self.weights["W" + str(i)] = np.random.randn(layers[i - 1],
                                                         layer_size) * he_et_al
            self.weights["b" + str(i)] = np.zeros((layers[i - 1], 1))
            layer_size = layers[i - 1]