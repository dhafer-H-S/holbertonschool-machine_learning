#!/usr/bin/env python3
"""
function trains a loaded neural network model using mini batch gradient descent
"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data
"""
function trains a loaded neural network model using mini batch gradient descent
"""


def create_mini_batches(X, Y, batch_size):
    """
    Creates mini-batches from the given dataset.

    Args:
        X (numpy.ndarray): The input data of shape (m, nx)
        where m is the number of data points and nx is
        the number of features.
        Y (numpy.ndarray): The labels of shape (m, ny)
        where m is the number of data points and ny is
        the number of classes.
        batch_size (int): The size of each mini-batch.

    Returns:
        list: A list of mini-batches, where each mini-batch
        is a tuple containing the input data and labels.
    """
    x_shuffled, y_shuffled = shuffle_data(X, Y)
    m = X.shape[0]
    mini_batches = list()
    if batch_size > m:
        batch_size = m
    for i in range(0, m, batch_size):
        X_batch = x_shuffled[i:i + batch_size]
        Y_batch = y_shuffled[i:i + batch_size]
        mini_batches.append((X_batch, Y_batch))
    return mini_batches
