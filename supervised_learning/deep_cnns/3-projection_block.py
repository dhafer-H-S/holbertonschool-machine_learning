#!/usr/bin/env python3
""" projection of block as described """


import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    F11, F3, F12 = filters
    """ convolution 1"""
    c1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(
        s, s), padding='same', kernel_initializer='he_normal')(A_prev)
    """batch normalisation 1"""
    b1 = K.layers.BatchNormalization(axis=3)(c1)
    """relu activation function 1"""
    a1 = K.layers.Activation('relu')(b1)
    """ convolution 2"""
    c2 = K.layers.Conv2D(
        filters=F3, kernel_size=(
            3, 3), strides=(
            1, 1), padding='same', kernel_initializer='he_normal')(a1)
    """batch normalisation 2"""
    b2 = K.layers.BatchNormalization(axis=3)(c2)
    """relu activation function 2"""
    a2 = K.layers.Activation('relu')(b2)
    """ convolution 3"""
    c3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(1, 1),
                         padding='same', kernel_initializer='he_normal')(a2)
    """batch normalisation 3"""
    b3 = K.layers.BatchNormalization(axis=3)(c3)
    """ convolution 4"""
    c4 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(
        s, s), padding='same', kernel_initializer='he_normal')(A_prev)
    """batch normalisation 4"""
    b4 = K.layers.BatchNormalization(axis=3)(c4)
    """addition of result after batching convolution 3 and convolution 4"""
    add = K.layers.Add()([b3, b4])
    """relu activation function for the output"""
    output = K.layers.Activation('relu')(add)
    return output
