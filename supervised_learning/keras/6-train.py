#!/usr/bin/env python3
"""train a model using mini batch gradient descent and analyse validation data and use early stopping"""
import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        validation_data=None,
        early_stopping=False,
        patience=0,
        verbose=True,
        shuffle=False):
    """
    network is the model to be trained
    data conating the inpute data of sahpe (m, nx)
    labels conating the labels of data of shape (m, classes)
    batch_size is the size of the batch used for mini batch gradient descent
    validation data is the data to validate the model with
    early stoping is a boolean that indicates whether early stopping should be used
    early stopping should only be performed if validation_data exists
    early stopping should be based on validation loss
    patience is the patience used for early stopping
    verbose is a bloolean that deteminess id output should be
    printed during training
    shuffle is a boolean that detemines whether to shuffle the batches
    every epoch normaly it's a good idea
    """
    callbacks = []
    if validation_data and early_stoping:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor=early_stopping, patience=patience)
        callbacks.append(early_stopping_callback)
    model = network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        validation_data=validation_data,
        shuffle=shuffle,
        callbacks=callbacks
    )
    return model
