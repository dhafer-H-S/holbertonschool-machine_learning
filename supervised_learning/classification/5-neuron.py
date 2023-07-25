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

    """calculates one pass of gradient descent on the neuron """

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ X is the input data """
        """ Y is the coorect labes for the input data"""
        """ A is the activated output of the neuron """

        """ alpha is the time gonna take the algorithms to learn from data """

        """contains the correct labels for the input data"""
        """we are using Y not A because we need the correct labels"""
        """ not the activated output of the neuron """
        m = Y.shape[1]

        """ calculating the diffrence between """
        """ the prediction output A and the correct output Y """

        """this is used for getting the diffrence and if everything is """
        """ alright or not the changes needed to the weight to reduce lose """
        dZ = A - Y

        """dW  represent gradient of the cost with respecting the weights"""
        """ dZ.T is the diffrence betwwen A and Y but in transposed way """
        """np.dot for multiplication of X and dZ.T"""
        dW = (1 / m) * np.dot(X, dZ.T)

        """db  represent gradient of the cost with respecting the bias"""
        """np.sum for calculating the sum of all elements in dZ"""
        db = (1 / m) * np.sum(dZ)

        """dW.T is the transpose for the gradien of the cost"""
        """taking transpose is necessary for  matching the sahpe of weight"""

        """alpha * dW.T scales the gradien form the learning rate"""

        """ the substraction updates the weights in the opposite direction"""
        """ of the gradient to minimize the cost function"""
        self.__W = self.__W - alpha * dW.T

        """ the substraction updates the bias in the opposite direction"""
        """ of the gradient to minimize the cost function"""
        self.__b = self.__b - alpha * db

        """ REMINDER """
        """ by updating the weight and the bias """
        """ the neuron will learn to adjust it's parameters"""
        """ in the direction to reduce the cost  """
        """ and improves the accuracy of it's predictions """
