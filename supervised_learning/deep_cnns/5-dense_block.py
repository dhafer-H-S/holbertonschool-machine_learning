#!/usr/bin/env python3
"""build a dense block """
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ build a dense block """
    for i in range(layers):
        batch_normalization = K.layers.BatchNormalization()(X)
        activation = K.layers.Activation('relu')(batch_normalization)
        conv2d = K.layers.Conv2D(filters=4 * growth_rate, kernel_size=1,
                                 padding='same',
                                 kernel_initializer='he_normal')(activation)
        batch_normalization = K.layers.BatchNormalization()(conv2d)
        activation = K.layers.Activation('relu')(batch_normalization)
        conv2d = K.layers.Conv2D(filters=growth_rate, kernel_size=3,
                                 padding='same',
                                 kernel_initializer='he_normal')(activation)
        X = K.layers.concatenate([X, conv2d])
        nb_filters += growth_rate
    return X, nb_filters
