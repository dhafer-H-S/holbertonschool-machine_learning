#!/usr/bin/env python3

""" DeepNeuralNetwork """

""" class of a DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    def __init__(self, nx, layers):
        """
        nx is the number of inpute featurs for this deep neural network
        layers is the number of nodes in each layer of the network 


        """
        if not isinstance(nx , int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        if not isinstance(layers, list) or len(layers) == 0 :
            raise TypeError("layers must be a list of positive integers")

        """ initialize weight and bias using he et al and bias intiated to 0 """
        self.L = len(layers) # number of layer in neural network
        self.cache = {} # dictionary to store intermediary values
        self.weights = {} # dictionary to store weight and biases
        """
        To initialize the weights in a layer using He et al.
        initialization with a a normal distribution as well by setting 
        µ = 0 
        and 
        sigma = np.sqrt{2/ F_{in}}
        where Fin is the number of input units in the layer
        """

        for l in range(1, self.L):
            if not all(isinstance(l, int) and l > 0 for l in layers):
                raise TypeError("layers must be a list of positive integers")
            # he et al weight initialization
            self.weights['w' + str(l)] = np.random.randn(layers[l - 1], layers[l]) * np.sqrt(2 / layers[l - 1])

            # zero bias initialization
            self.weights['b' + str(l)] = np.zeros((1, layers[l]))

        # initialize the last layer weight and biases
        self.weights['w' + str(self.L)] = np.random.randn(layers[self.L - 1], 1) * np.sqrt(2 / layers[self.L - 1])

        self.weights['b' + str(self.L)] = np.zeros((1, 1))

        


