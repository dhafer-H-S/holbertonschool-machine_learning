#!/usr/bin/env python3

""" DeepNeuralNetwork """

""" class of a DeepNeuralNetwork """
import numpy as np

class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """
        Initializes a deep neural network performing binary classification.

        Parameters:
        nx (int): Number of input features.
        layers (list): List representing the number of nodes in each layer of the network.

        Raises:
        TypeError: If nx is not an integer or layers is not a list.
        ValueError: If nx is less than 1.
        TypeError: If layers is an empty list or contains non-positive integers.

        Attributes:
        L (int): The number of layers in the neural network.
        cache (dict): A dictionary to hold all intermediary values of the network.
        weights (dict): A dictionary to hold all weights and biases of the network.
        """
        # Check if nx is an integer and positive
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        # Check if layers is a list and contains positive integers
        if not isinstance(layers, list) or not layers :
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers) # number of layers in the neural network
        self.cache = {} # dictionary to store intermediary values
        self.weights = {} # dictionary to store weights and biases

        # Initialize weights and biases using He et al. method and zeros
        for l in range(1, self.L + 1):
            # He et al. weight initialization
            self.weights['W' + str(l)] = np.random.randn(layers[l - 1], nx) * np.sqrt(2 / nx)
            
            # Zero bias initialization
            self.weights['b' + str(l)] = np.zeros((layers[l - 1], 1))

        # # Initialize the last layer weights and biases
        # self.weights['W' + str(self.L)] = np.random.randn(layers[self.L - 1], 1) * np.sqrt(2 / layers[self.L - 1])
        # self.weights['b' + str(self.L)] = np.zeros((1, 1))
