#!/usr/bin/env python3

""" DeepNeuralNetwork """

""" class of a DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        if not isinstance(layers, list) or len(layers) == 0 or not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers) # number of layers in the neural network
        self.cache = {} # dictionary to store intermediary values
        self.weights = {} # dictionary to store weights and biases

        # Initialize weights and biases using He et al. method and zeros
        for l in range(1, self.L + 1):
            if l == 1:
                # Input layer, no He et al. method used
                self.weights['W' + str(l)] = np.random.randn(nx, layers[l - 1]) * np.sqrt(2 / nx)
            else:
                # Hidden layers and output layer use He et al. method
                self.weights['W' + str(l)] = np.random.randn(layers[l - 2], layers[l - 1]) * np.sqrt(2 / layers[l - 2])

            self.weights['b' + str(l)] = np.zeros((1, layers[l - 1]))