#!/usr/bin/env python3

""" DeepNeuralNetwork """

""" class of a DeepNeuralNetwork """
import numpy as np

class DeepNeuralNetwork:
    # deff methode to initializa the deep neural network
    def __init__(self, nx, layers):
        # check conditions 
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        # set public instance attributes
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        
        # loop to iterates through the range of numbers of layers 
        for l in range(self.L):
            # check if the elements in the layers are integers or not 
            # and check if theyeare greater then 0 or not 
            if not isinstance(layers[l], int) or layers[l] <= 0:
                raise TypeError("layers must be a list of positive integers")
            # initialise the weight if we are working on the first layer we 
            # gonna use He et al as intialize methode to generate weights with
            # a shape based on layers(l) and nx ( inputes features ) and then 
            # scale them usnig square root 2 / nx
            if l == 0:
                self.weights['W' + str(l + 1)] = np.random.randn(layers[l], nx) * np.sqrt(2 / nx)
            else:
                #for layers other than the first one it genrates random weigths
                # with a shape layers(l) - layers(l - 1) and scale them using
                # the square root of 2 / layers(l - 1)
                self.weights['W' + str(l + 1)] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2 / layers[l - 1])
            # initialize the biases for the current layer with zeros and store
            # them in the weights
            self.weights['b' + str(l + 1)] = np.zeros((layers[l], 1))
