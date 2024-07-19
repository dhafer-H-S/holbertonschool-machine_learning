#!/usr/bin/env python3
"""Identity Block"""
from tensorflow import keras as K


def identity_block(A_prev, filters):
    """identity_block Builds an identity block"""
    F11, F3, F12 = filters
    c1 = K.layers.Conv2D(
        filters=F11, kernel_size=(
            1, 1), padding='same', strides=(
            1, 1), kernel_initializer=K.initializers.he_normal(
                seed=0))(A_prev)
    b1 = K.layers.BatchNormalization(axis=3)(c1)
    r1 = K.layers.Activation('relu')(b1)

    # Second component of main path
    c2 = K.layers.Conv2D(
        filters=F3, kernel_size=(
            3, 3), padding='same', strides=(
            1, 1), kernel_initializer=K.initializers.he_normal(
                seed=0))(r1)
    b2 = K.layers.BatchNormalization(axis=3)(c2)
    r2 = K.layers.Activation('relu')(b2)

    # Third component of main path
    c3 = K.layers.Conv2D(
        filters=F12, kernel_size=(
            1, 1), padding='same', strides=(
            1, 1), kernel_initializer=K.initializers.he_normal(
                seed=0))(r2)
    b3 = K.layers.BatchNormalization(axis=3)(c3)

    # Final step: Add shortcut value to main path, and pass it through a RELU
    # activation
    A = K.layers.Add()([b3, A_prev])
    A = K.layers.Activation('relu')(A)

    return A
