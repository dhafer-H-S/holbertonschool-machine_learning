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
        for l in range(1, self.__L + 1):
            """ Initialize weights using He et al. initialization method """
            he_et_al = np.sqrt(2 / layer_size)
            self.__weights["W" + str(l)] = np.random.randn(
                layers[l - 1], layer_size) * he_et_al
            self.__weights["b" + str(l)] = np.zeros((layers[l - 1], 1))
            layer_size = layers[l - 1]

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
        for l in range(1, self.__L + 1):
            """ Get data, weight, and bias """
            data = self.__cache['A' + str(l - 1)]
            w = self.__weights['W' + str(l)]
            bias = self.__weights['b' + str(l)]

            """ Perform forward propagation """
            Z = np.dot(w, data) + bias
            A = 1 / (1 + np.exp(-Z))

            """ Store data in cache """
            self.__cache['Z' + str(l)] = Z
            self.__cache['A' + str(l)] = A

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
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]

        dz = self.__cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_prev = self.__cache['A' + str(i - 1)]
            dw = 1/m * np.dot(dz, A_prev.T)
            db = 1/m * np.sum(dz, axis=1, keepdims=True)
            dz = np.dot(self.__weights['W' + str(i)].T,
                        dz) * A_prev * (1 - A_prev)
            self.__weights['W' + str(i)] -= alpha * dw
            self.__weights['b' + str(i)] -= alpha * db


    """ def methode train to train the model """
    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ X contain the inpute data """
        """ Y contain the correct labels for the inpute data """
        """ numberof iteration to train over the model """
        """ learning rate """
        m = Y.shape[1]
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
        return self.evaluate(X, Y)
