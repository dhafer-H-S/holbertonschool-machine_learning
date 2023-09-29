#!/usr/bin/env python3
"""
train a model using mini batch gradient descent and analyse validation data
and use early stopping and also train with learining rate decay
"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs, validation_data=None, early_stopping=False, patience=0, learning_rate_decay=False, alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
    """
    network is the model to be trained
    data containing the input data of shape (m, nx)
    labels containing the labels of data of shape (m, classes)
    batch_size is the size of the batch used for mini-batch gradient descent
    validation_data is the data to validate the model with
    early_stopping is a boolean that indicates whether early stopping should be used
    early stopping should only be performed if validation_data exists
    early stopping should be based on validation loss
    patience is the patience used for early stopping
    learning_rate_decay is a boolean that indicates whether learning rate decay should be used
    learning rate decay should only be performed if validation_data exists
    the decay should be performed using inverse time decay
    the learning rate should decay in a stepwise fashion after each epoch
    each time the learning rate updates, Keras should print a message
    alpha is the initial learning rate
    decay_rate is the decay rate
    verbose is a boolean that determines if output should be printed during training
    shuffle is a boolean that determines whether to shuffle the batches every epoch
    """
    callbacks = []
    if validation_data and early_stopping:
        early_stopping_callback = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stopping_callback)

    if validation_data and learning_rate_decay:
        def schedule(epoch):
            return alpha / (1 + decay_rate * epoch)

        lr_scheduler_callback = K.callbacks.LearningRateScheduler(
            schedule,
            verbose=1
        )

        class LrUpdaterCallback(K.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                new_lr = schedule(epoch)
                K.backend.set_value(self.model.optimizer.lr, new_lr)
                print(f'Learning rate updated to {new_lr:.5f}')

        lr_updater_callback = LrUpdaterCallback()

        callbacks.append(lr_scheduler_callback)
        callbacks.append(lr_updater_callback)

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
