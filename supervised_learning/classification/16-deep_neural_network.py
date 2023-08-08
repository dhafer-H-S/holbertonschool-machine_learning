#!/usr/bin/env python3

""" DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """ class of a DeepNeuralNetwork """

    """ deff methode to initializa the deep neural network """
    def __init__(self, nx, layers):
        """ check conditions """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")

        """
        check if the elements in the layers are integers or not
        and check if theye are greater then 0 or 
        """

        """ set public instance attributes """
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        layer_size = nx

        """ loop to iterates through the range of numbers of layers """
        """
            initialise the weight if we are working on the first layer we
            +gonna use He et al as intialize methode to generate weights with
            a shape based on layers(l) and nx ( inputes features ) and then
            scale them usnig square root 2 / nx
        """

        """
            initialize the biases for the current layer with zeros and store
            them in the weights
        """
        for l in range(1, self.L + 1):

            he_et_al = np.sqrt(2 / layer_size)
            self.weights["W" + str(l)] = np.random.randn(layers[l - 1],
                                                         layer_size) * he_et_al
            self.weights["b" + str(l)] = np.zeros((layers[l - 1], 1))
            layer_size = layers[l - 1]
