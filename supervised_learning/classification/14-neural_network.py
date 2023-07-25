#!/usr/bin/env python3
""" neural network """
import numpy as np


class NeuralNetwork:
    """a class for the neural network from scratch"""

    # initializing everything in the neural network

    def __init__(self, nx, nodes):  # a public method iinitialization
        # nx is the number of inpute data
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx < 1:
            raise ValueError("nx must be a positive integer")
        # nodes is the number of nodes found in the hiden layer
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes < 1:
            raise ValueError("nodes must be a positive integer")

        """ the weights vector for the hiden layer"""
        """ initialisied withe a random normal distribution value"""
        """ w1 is the connection between the inpute and the hiden layer """
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        """ we initialize b1 as an array of zeros with shape(nodes, a)"""
        """that allows the neurol network to learn from biases for each node"""
        self.__A1 = 0
        """ w2  is the connection between the hiden layer and the output"""
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter for w1"""
        return self.__W1

    @property
    def b1(self):
        """getter for b1"""
        return self.__b1

    @property
    def A1(self):
        """getter for A1"""
        return self.__A1

    @property
    def W2(self):
        """getter for w2"""
        return self.__W2

    @property
    def b2(self):
        """getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """getter for A2"""
        return self.__A2
    """ forward propagation methode """

    def forward_prop(self, X):
        """ X contains the input data """
        """ calculate from input to hiden layers """
        z1 = np.dot(self.__W1, X) + self.__b1
        """ canculating A1 activation function sigmoid"""
        self.__A1 = 1 / (1 + np.exp(-z1))
        """ calculate from hiden layer to output"""
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        """ canculating A2 activation function sigmoid"""
        self.__A2 = 1 / (1 + np.exp(-z2))

        return self.__A1, self.__A2

    """ loss or cost function methode """

    def cost(self, Y, A):
        """ Y conatin the correct labels for the input data """
        """ A contain the activated output of the neuron """
        """ m is number of exapmles in the input data"""

        m = Y.shape[1]

        """ cost function using logistic regression """
        cost = -(1 / m) * np.sum(Y * np.log(A) +
                                 (1 - Y) * np.log(1.0000001 - A))
        return cost

    """ evaluate the neural network """

    def evaluate(self, X, Y):
        """ X contain the inpute data """
        """ Y contain the correct labels for the inpute data """

        """ m is number of exapmles in the input data"""
        m = Y.shape[1]
        """forward propagation function to find __A2 for the prediction """
        self.forward_prop(X)
        prediction = np.where(self.__A2 >= 0.5, 1, 0)
        """ cost function calculation"""
        cost = self.cost(Y, self.__A2)
        """ returning the values """
        return prediction, cost

    """ gradient descent function """

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ X contain the input data """
        """ Y contain the correct labeled data """
        """ alpha learning rate"""

        """ calculate gradient for the output layer """
        m = Y.shape[1]
        dA2 = (A2 - Y) / m
        dW2 = np.dot(dA2, A1.T)
        db2 = np.sum(dA2, axis=1, keepdims=True)

        """ calculate gradient for the hiden layer """
        dA1 = np.dot(self.__W2.T, dA2) * (A1 * (1 - A1))
        dW1 = np.dot(dA1, X.T)
        db1 = np.sum(dA1, axis=1, keepdims=True)

        """ update the paramters """
        self.__W2 = self.__W2 - (alpha * dW2)
        self.__W1 = self.__W1 - (alpha * dW1)
        self.__A2 = self.__A2 - (alpha * dA2)
        self.__A1 = self.__A1 - (alpha * dA1)

    """ train function methode """

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ X contain the input data """
        """ Y contain the correct labels for the inpute data """
        """ alpha is the training rate  """
        """ iteration is the number of iteration to train the model over """
        m = Y.shape[1]
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)

        prediction, cost = self.evaluate(X, Y)
        return prediction, cost
