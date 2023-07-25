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
