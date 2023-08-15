#!/usr/bin/env python3

""" DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """ class of a DeepNeuralNetwork """

    """ deff methode to initializa the deep neural network """

    def __init__(self, nx, layers):
        """nx is the inpute features in the neural network"""
        """ layers is the number of nodes in each layer of the network"""

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
        and check if theye are greater then 0 or not
        """

        """ set private instance attributes """
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
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
        for l in range(1, self.__L + 1):

            he_et_al = np.sqrt(2 / layer_size)
            self.__weights["W" + str(l)] = np.random.randn(
                layers[l - 1], layer_size) * he_et_al
            self.__weights["b" + str(l)] = np.zeros((layers[l - 1], 1))
            layer_size = layers[l - 1]

    """ getters for private attributes """
    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    """ def methode for froward propagation for the neural netwok"""
    def forward_prop(self, X):
        """X should be saved to the cache dictionary using the key A0"""
        self.__cache['A0'] = X
        """ loop for calculating through every layer in the neural network"""
        for l in range(self.__L):
            """ set the methode to get data """
            data = self.__cache['A' + str(l)]
            """ set methode to get the weight"""
            w = self.__weights['W' + str(l + 1)]
            """set the methode to get the bias """
            bias = self.__weights['b' + str(l + 1)]
            """ calculation the froword propagation """
            Z = np.dot(w, data) + bias
            """ calcualte the activation function using sigmoid"""
            A = 1 / (1 + np.exp(-Z))
            """ store data in cache"""
            self.__cache['A' + str(l + 1)] = A
        return A, self.__cache
    """ def methode for cost function"""
    def cost(self, Y, A):
        """ y contain the correct labels for the input data """
        """ A contain the acivated output for the neuron for eache example """
        m = Y.shape[1]
        cost = - ( 1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost