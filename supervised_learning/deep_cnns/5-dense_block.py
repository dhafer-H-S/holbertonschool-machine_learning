#!/usr/bin/env python3
"""build a dense block """
from tensorflow import keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ build a dense block """
    for _ in range(layers):
        batch_norm = layers.BatchNormalization()(X)
        relu = layers.Activation('relu')(batch_norm)
        conv1x1 = layers.Conv2D(filters=4 * growth_rate, kernel_size=1, padding='same', kernel_initializer='he_normal')(relu)
        batch_norm = layers.BatchNormalization()(conv1x1)
        relu = layers.Activation('relu')(batch_norm)
        conv3x3 = layers.Conv2D(filters=growth_rate, kernel_size=3, padding='same', kernel_initializer='he_normal')(relu)
        X = layers.concatenate([X, conv3x3])
        nb_filters += growth_rate

    return X, nb_filters
