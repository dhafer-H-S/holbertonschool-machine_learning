#!/usr/bin/env python3
"""train a model using mini batch gradient descent"""
import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        verbose=True,
        shuffle=False):
    """
    network is the model to be trained
    data conating the inpute data of sahpe (m, nx)
    labels conating the labels of data of shape (m, classes)
    batch_size is the size of the batch used for mini batch gradient descent
    verbose is a bloolean that deteminess id output should be
    printed during training
    shuffle is a boolean that detemines whether to shuffle the batches
    every epoch normaly it's a good idea
    """
    model = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
    )
    return model
