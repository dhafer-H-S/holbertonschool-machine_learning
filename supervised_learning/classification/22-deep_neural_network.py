#!/usr/bin/env python3
"""deep neural network performing binary classification"""


import numpy as np


class DeepNeuralNetwork:
    """deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for al in range(len(layers)):
            if type(layers[al]) != int or layers[al] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if al == 0:
                He = np.random.randn(layers[al], nx) * np.sqrt(2 / nx)
                self.__weights['W' + str(al + 1)] = He
            else:
                He = np.random.randn(
                    layers[al], layers[al - 1]) * np.sqrt(2 / layers[al - 1])
                self.__weights['W' + str(al + 1)] = He

            self.__weights['b' + str(al + 1)] = np.zeros((layers[al], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neuron'''
        self.__cache["A0"] = X
        for i in range(self.__L):
            z = np.dot(self.__weights['W' + str(i + 1)],
                       self.__cache['A'+str(i)]) +\
                self.__weights['b'+str(i + 1)]
            A = 1 / (1 + np.exp(-z))
            ''''sigmoid activation function'''
            self.__cache['A' + str(i + 1)] = A
        return A, self.__cache

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression'''
        m = Y.shape[1]
        cost = (-1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1.0000001-A))
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the deep neural network’s predictions'''
        A, _ = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        cost = self.cost(Y, A)
        return prediction, cost

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

    def train(self, X, Y, iterations=5000, alpha=0.05):
        '''Trains the neural network'''
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