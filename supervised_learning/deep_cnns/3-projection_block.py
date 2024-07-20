#!/usr/bin/env python3
"""Projection of block as described"""
from tensorflow import keras as K


def projection_block(A_prev, filters, s=2):
    """Projection block for resnet."""
    F11, F3, F12 = filters
    init = K.initializers.he_normal(seed=0)  
    """Using the same initializer with a seed for reproducibility"""

    """Convolution 1"""
    c1 = K.layers.Conv2D(
        filters=F11, kernel_size=(
            1, 1), strides=(
            s, s), padding='same', kernel_initializer=init)(A_prev)
    """Batch Normalization 1"""
    b1 = K.layers.BatchNormalization(axis=3)(c1)
    """ReLU Activation 1"""
    a1 = K.layers.Activation('relu')(b1)

    """Convolution 2"""
    c2 = K.layers.Conv2D(
        filters=F3, kernel_size=(
            3, 3), strides=(
            1, 1), padding='same', kernel_initializer=init)(a1)
    """Batch Normalization 2"""
    b2 = K.layers.BatchNormalization(axis=3)(c2)
    """ReLU Activation 2"""
    a2 = K.layers.Activation('relu')(b2)

    """Convolution 3"""
    c3 = K.layers.Conv2D(
        filters=F12, kernel_size=(
            1, 1), strides=(
            1, 1), padding='same', kernel_initializer=init)(a2)
    """Batch Normalization 3"""
    b3 = K.layers.BatchNormalization(axis=3)(c3)

    """Convolution 4 for shortcut connection"""
    c4 = K.layers.Conv2D(
        filters=F12, kernel_size=(
            1, 1), strides=(
            s, s), padding='same', kernel_initializer=init)(A_prev)
    """Batch Normalization 4 for shortcut connection"""
    b4 = K.layers.BatchNormalization(axis=3)(c4)

    """Adding the shortcut to the main path"""
    add = K.layers.Add()([b3, b4])
    """Final ReLU Activation"""
    output = K.layers.Activation('relu')(add)
    return output
