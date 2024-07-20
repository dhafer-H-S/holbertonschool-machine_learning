#!/usr/bin/env python3
""" transition layer"""


from tensorflow import keras as K

def transition_layer(X, nb_filters, compression):
    """transition layer 1 × 1 conv followed by 2 × 2 average pool, stride 2"""
    init = K.initializers.he_normal(seed=None)
    filters = int(nb_filters * compression)
    layer = K.layers.BatchNormalization()(X)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.Conv2D(filters=filters,
                            kernel_size=1,
                            padding='same',
                            kernel_initializer=init)(layer)
    layer = K.layers.AveragePooling2D(pool_size=2,
                                      strides=2,
                                      padding='same')(layer)
    return layer, filters
