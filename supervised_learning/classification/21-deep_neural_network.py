#!/usr/bin/env python3

""" DeepNeuralNetwork """
import numpy as np


class DeepNeuralNetwork:
    """ class of a DeepNeuralNetwork """

    """ def method to initialize the deep neural network """

    def __init__(self, nx, layers):
        """ nx is the number of input features in the neural network """
        """ layers is the list of the number of nodes in
        each layer of the network """

        """ Check conditions """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        if not all(map(lambda x: x > 0 and isinstance(x, int), layers)):
            raise TypeError("layers must be a list of positive integers")

        """ Set private instance attributes """
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        layer_size = nx

        """ Loop through the range of numbers of layers """
        for layer_index in range(1, self.L + 1):
            he_et_al = np.sqrt(2 / layer_size)
            self.weights["W" + str(layer_index)] = np.random.randn(
                layers[layer_index - 1], layer_size) * he_et_al
            self.weights["b" + str(layer_index)
                         ] = np.zeros((layers[layer_index - 1], 1))
            layer_size = layers[layer_index - 1]

    """ Getters for private attributes """
    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    """ Method for forward propagation of the neural network """

    def forward_prop(self, X):
        """ X should be saved to the cache dictionary using the key A0 """
        self.__cache['A0'] = X
        self.__cache['Z0'] = X

        """ Loop through every layer in the neural network """
        for layer_index in range(self.__L):
            # set the method to get data
            data = self.__cache['A' + str(layer_index)]
            # set method to get the weight
            w = self.__weights['W' + str(layer_index + 1)]
            # set the method to get the bias
            bias = self.__weights['b' + str(layer_index + 1)]
            # calculation the forward propagation
            Z = np.dot(w, data) + bias
            # calculate the activation function using sigmoid
            A = 1 / (1 + np.exp(-Z))
            # store data in cache
            self.__cache['A' + str(layer_index + 1)] = A
        return A, self.__cache

    """ Method for cost function """

    def cost(self, Y, A):
        """ Calculate the cost function """
        m = Y.shape[1]
        cost = (-1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    """ Method to evaluate the deep neural network and its predictions """

    def evaluate(self, X, Y):
        """ Perform forward propagation to calculate predictions and cost """
        m = X.shape[1]
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)

        """ Make predictions """
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    """ Method for gradient descent to train the neural network """

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculate gradients and update weights and biases """
        m = Y.shape[1]
        """ calculate initiale derivation of cost with respect to activations """
        dA = - (np.divide(Y, cache['A' + str(self.__L)]) -
                np.divide(1 - Y, 1 - cache['A' + str(self.__L)]))

        for l in range(self.__L, 0, -1):
            Z = cache['Z' + str(l)]
            A_prev = cache['A' + str(l - 1)]
            W = self.__weights['W' + str(l)]
            """ calculate gradient for weights and biases """
            dZ = dA * cache['A' + str(l)] * (1 - cache['A' + str(l)])
            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m
            """ propagate error to the previous layer """
            dA = np.dot(W.T, dZ)
            """ update weights and biases using the learning rate alpha """
            self.__weights['W' + str(l)] -= alpha * dW
            self.__weights['b' + str(l)] -= alpha * db
