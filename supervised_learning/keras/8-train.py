#!/usr/bin/env python3
"""Train keras Model."""
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
        learning_rate_decay=False,
        alpha=0.1,
        decay_rate=1,
        save_best=False,
        filepath=None,
        verbose=True,
        shuffle=False):
    """
    network is the model to be trained
    data containing the input data of shape (m, nx)
    labels containing the labels of data of shape (m, classes)
    batch_size is the size of the batch used for mini-batch gradient descent
    validation_data is the data to validate the model with
    early_stopping is a boolean that indicates whether early stopping
    should be used
    early stopping should only be performed if validation_data exists
    early stopping should be based on validation loss
    patience is the patience used for early stopping
    learning_rate_decay is a boolean that indicates whether learning
    rate decay should be used
    learning rate decay should only be performed if validation_data exists
    the decay should be performed using inverse time decay
    the learning rate should decay in a stepwise fashion after each epoch
    each time the learning rate updates, Keras should print a message
    alpha is the initial learning rate
    decay_rate is the decay rate
    verbose is a boolean that determines if output should be printed
    during training
    shuffle is a boolean that determines whether to shuffle the batches
    every epoch
    save best is a boolean indicating wheter to save the model after
    each epoch if it is the best , a model is considered the best if it's
    validation loss is the lowest that the model has obtained
    filepath is where the model should be saved
    """
    """
    Returns: the History object generated after training the model
    """
    def schedule(epoch):
        previous_lr = 1

        def lr(epoch, start_lr, decay):
            nonlocal previous_lr
            previous_lr *= (start_lr / (1. + decay * epoch))
            return previous_lr
        return lr(epoch, alpha, decay_rate)

    callbacks = []
    if validation_data:
        if early_stopping:
            early_stopping_callback = K.callbacks.EarlyStopping(
                'val_loss', patience=patience)
            callbacks.append(early_stopping_callback)
        if learning_rate_decay:
            lr_callback = K.callbacks.LearningRateScheduler(
                schedule, verbose=True)
            callbacks.append(lr_callback)
        if save_best and filepath:
            save_best_callback = K.callbacks.ModelCheckpoint(
                filepath,
                save_best_only=save_best
            )
            callbacks.append(save_best_callback)
    history = network.fit(data, labels, batch_size, epochs,
                          verbose=verbose, shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return history
