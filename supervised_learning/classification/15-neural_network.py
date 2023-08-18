#!/usr/bin/env python3
"""neural network with one hidden layer performing binary classification"""


import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """neural network with one hidden layer performing binary classification"""

    def __init__(self, nx, nodes):
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) != int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        '''Calculates the forward propagation of the neuron'''
        z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-z1))
        z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-z2))
        ''''sigmoid activation function'''
        return self.__A1, self.__A2

    def cost(self, Y, A):
        '''Calculates the cost of the model using logistic regression'''
        m = Y.shape[1]
        cost = (-1/m) * np.sum(Y*np.log(A) + (1-Y)*np.log(1.0000001-A))
        return cost

    def evaluate(self, X, Y):
        '''Evaluates the neural network’s predictions'''
        self.forward_prop(X)
        prediction = np.where(self.A2 >= 0.5, 1, 0)
        cost = self.cost(Y, self.A2)
        return prediction, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network"""
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = (1 / m) * np.dot(dZ2, A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.dot(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = (1 / m) * np.dot(dZ1, X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1
        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        '''Trains the neuron'''
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if type(step) != int:
            raise TypeError("step must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if step < 1 or step > iterations:
            raise ValueError("step must be positive and <= iterations")


        costs = []
        for i in range(0, iterations + 1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            costs.append(self.cost(Y, self.__A2))
            if verbose:
                print("Cost after ",i ," iterations: ",self.cost(Y, self.A2))
            if i != 0 and graph and (i % step) == 0:
                plt.plot(costs)
                plt.xlabel('iteration')
                plt.ylabel('costs')
                plt.title('Training Cost')
                plt.show()
        return self.evaluate(X, Y)