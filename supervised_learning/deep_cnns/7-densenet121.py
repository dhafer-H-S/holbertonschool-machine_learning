#!/usr/bin/env python3
""" DenseNet-121 """


from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """desnet121 function"""
    init = K.initializers.he_normal(seed=None)
    X = K.Input(shape=(224, 224, 3))
    layer = K.layers.BatchNormalization()(X)
    layer = K.layers.Activation('relu')(layer)
    layer = K.layers.Conv2D(filters=2 * growth_rate,
                            kernel_size=7,
                            padding='same',
                            strides=2,
                            kernel_initializer=init)(layer)
    layer = K.layers.MaxPooling2D(pool_size=3,
                                  strides=2,
                                  padding='same')(layer)
    layer, nb_filters = dense_block(layer, 2 * growth_rate, growth_rate, 6)
    layer, nb_filters = transition_layer(layer, nb_filters, compression)
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, 12)
    layer, nb_filters = transition_layer(layer, nb_filters, compression)
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, 24)
    layer, nb_filters = transition_layer(layer, nb_filters, compression)
    layer, nb_filters = dense_block(layer, nb_filters, growth_rate, 16)
    layer = K.layers.AveragePooling2D(pool_size=7,
                                      strides=None,
                                      padding='same')(layer)
    layer = K.layers.Dense(units=1000,
                           activation='softmax',
                           kernel_initializer=init)(layer)
    model = K.models.Model(inputs=X, outputs=layer)
    return model
