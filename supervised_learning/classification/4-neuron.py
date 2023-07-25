#!/usr/bin/env python3
"""supervised neuron file that recognise numbers using some trained data"""
import numpy as np


class Neuron:
    """class for supervised neuron"""

    def __init__(self, nx):
        """initialisation of the supervised neuron"""
        """nX number of input feature in neuron """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        elif nx < 1:
            raise ValueError("nx must be a positive integer")

        """declaring private instance attribute"""
        """ the weight of neuron """
        self.__W = np.random.randn(1, nx)
        """b is for the bias of the neuron"""
        self.__b = 0
        """A is the activated output  of the neuron """
        self.__A = 0

    @property
    def W(self):
        """getter for W"""
        return self.__W

    @property
    def A(self):
        """getter for A"""
        return self.__A

    @property
    def b(self):
        """getter for b"""
        return self.__b

    """creating public method for forword propagation"""

    def forward_prop(self, X):
        """calculating the forward propagation for the neuron"""
        """perform matrix multiplication between __w and x"""
        Z = np.dot(self.__W, X) + self.__b

        """calcualtig the sigmoid activation function for neuron"""
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    """calculate the coast of a the model using logistic regression"""

    """creating a public methode to calculate the cost"""
    """cost is the same as loss function"""

    def cost(self, Y, A):
        """m is number of examples in inpute data"""
        m = Y.shape[1]
        """ we use log to stabilise calculation specially with small values"""
        costvalue = -(1 / m) * np.sum(Y * np.log(A) +
                                      (1 - Y) * np.log(1.0000001 - A))
        return costvalue

    """creating a public methode for evaluate the neuron prediction"""

    def evaluate(self, X, Y):
        """X contains the inpute data"""
        # X = np.numpy.ndarray(shape=(self.nx, self.m))
        """nx is the number of inpute features of the data"""
        """m is the number of examples in the input data"""

        """Y contains the correct labels for the inputes data"""
        m = Y.shape[1]
        A = self.forward_prop(X)
        prediction = np.where(A >= 0.5, 1, 0)
        """we had to call for the A, and Y """
        """ Y the argument the correct labes for the inpute data"""
        """ A the activated outout of the neuron"""
        costvalue = self.cost(Y, A)
        return prediction, costvalue
