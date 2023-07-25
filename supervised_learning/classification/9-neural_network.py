#!/usr/bin/env python3
""" neural network """
import numpy as np


class NeuralNetwork:
    """a class for the neural network from scratch"""

    # initializing everything in the neural network

    def __init__(self, nx, nodes):  # a public method iinitialization
        # nx is the number of inpute data
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        # nodes is the number of nodes found in the hiden layer
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        """ the weights vector for the hiden layer"""
        """ initialisied withe a random normal distribution value"""
        """ w1 is the connection between the inpute and the hiden layer """
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        """ we initialize b1 as an array of zeros with shape(nodes, a)"""
        """that allows the neurol network to learn from biases for each node"""
        self.__A1 = 0
        """ w2  is the connection between the hiden layer and the output"""
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter for w1"""
        return self.__W1

    @property
    def b1(self):
        """getter for b1"""
        return self.__b1

    @property
    def A1(self):
        """getter for A1"""
        return self.__A1

    @property
    def W2(self):
        """getter for w2"""
        return self.__W2

    @property
    def b2(self):
        """getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """getter for A2"""
        return self.__A2
