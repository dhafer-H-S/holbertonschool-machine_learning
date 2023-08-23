#!/usr/bin/env python3

""" DeepNeuralNetwork """
import numpy as np
import matplotlib.pyplot as plt
import pickle


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

        """ Loop through every layer in the neural network """
        for l in range(1, self.__L + 1):
            """ Get data, weight, and bias """
            data = self.__cache['A' + str(l - 1)]
            w = self.__weights['W' + str(l)]
            bias = self.__weights['b' + str(l)]

            """ Perform forward propagation """
            Z = np.dot(w, data) + bias
            """ calculate activation function using softmax """
            if l == self.L:
                A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
                self.__cache['A' + str(l)] = A
            else:
                """ calculat eusing sigmoid function"""
                A = 1 / (1 + np.exp(-Z))

                """ Store data in cache """
                self.__cache['A' + str(l)] = A

        return A, self.__cache

    """ Method for cost function to Calculate the cross-entropy cost """

    def cost(self, Y, A):
        """ Y (numpy.ndarray): One-hot encoded target labels
        of shape (classes, m)
        """
        """ A (numpy.ndarray): Predicted activations of shape (classes, m)"""
        """ Calculate the cost function """
        if not isinstance(Y, np.ndarray) or len(Y.shape) != 2:
            return None
        m = Y.shape[1]
        cost = -1 / m * np.sum(Y * np.log(A))
        return cost

    """ Method to evaluate the deep neural network and its predictions """

    def evaluate(self, X, Y):
        """
        X (numpy.ndarray): Input data of shape (input features, m).
        Y (numpy.ndarray): One-hot encoded target labels of shape (classes, m).
        """
        A, cache = self.forward_prop(X)
        """  Convert softmax output to class predictions """
        predictions = np.argmax(A, axis=0)
        true_labels = np.argmax(Y, axis=0)
        accuracy = np.mean(predictions == true_labels)
        cost = self.cost(Y, A)
        return accuracy, cost

    """ Method for gradient descent to train the neural network """

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Number of training examples"""
        m = Y.shape[1]
        """ Calculate the initial derivative of the cost with
        respect to activations """
        dz = cache['A' + str(self.__L)] - Y
        for i in range(self.__L, 0, -1):
            A_prev = cache['A' + str(i - 1)]
            """ Calculate the gradients for weights and biases """
            dw = 1 / m * np.dot(dz, A_prev.T)
            db = 1 / m * np.sum(dz, axis=1, keepdims=True)
            """ Calculate the derivative of the pre-activation Z """
            dz = np.dot(self.__weights['W' + str(i)].T,
                        dz) * A_prev * (1 - A_prev)
            """ Update weights and biases using the learning rate alpha """
            self.__weights['W' + str(i)] -= alpha * dw
            self.__weights['b' + str(i)] -= alpha * db

    """ def methode train to train the model """

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ X contain the inpute data """
        """ Y contain the correct labels for the inpute data """
        """ numberof iteration to train over the model """
        """ learning rate """
        m = Y.shape[1]
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not isinstance(step, int):
            raise TypeError("step must be an integer")
        if step <= 0 or step > iterations:
            step = iterations
        costs = []
        iter = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            cost = self.cost(Y, A)
            if verbose and ((i % step) == 0 or i == iterations):
                print("Cost after", i, " iterations: ", cost)
                costs.append(cost)
                iter.append(i)

            

        if graph:
            plt.plot(iter, costs, 'b')
            plt.title("Training Cost")
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.show()

        return self.evaluate(X, Y)

    """ def function methode to save the instance objects to a file """
    def save(self, filename):
        """ add .pkl to the filename if it dosen't exist"""
        if not filename.endswith(".pkl"):
            filename += ".pkl"
        """ open file and write in it"""
        with open(filename, "wb") as file:
            """ save inside the file"""
            pickle.dump(self, file)

    """ def function methode to load data to file"""
    @staticmethod
    def load(filename):
        try:
            """open file and read it's content"""
            with open(filename, 'rb') as file:
                loaded_object = pickle.load(file)
            return loaded_object
        except FileNotFoundError:
            return None