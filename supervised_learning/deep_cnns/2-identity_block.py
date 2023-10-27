#!/usr/bin/env python3
"""Identity Block"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """identity_block Builds an identity block"""
    F11, F3, F12 = filters
    c1 = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), padding='same', strides=(1, 1), kernel_initializer='he_normal')(A_prev)
    b1 = K.layers.BatchNormalization(axis=3)(c1)
    r1 = K.layers.Activation('relu')(b1)
    c2 = K.layers.Conv2D(filters=F3, kernel_size=(3, 3), padding='same', strides=(1, 1), kernel_initializer='he_normal')(r1)
    b2 = K.layers.BatchNormalization(axis=3)(c2)
    r2 = K.layers.Activation('relu')(b2)
    c3 = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), padding='same', strides=(1, 1), kernel_initializer='he_normal')(r2)
    b3 = K.layers.BatchNormalization(axis=3)(c3)
    add = K.layers.Add()([b3, A_prev])
    output = K.layers.Activation('relu')(add)
    return output
