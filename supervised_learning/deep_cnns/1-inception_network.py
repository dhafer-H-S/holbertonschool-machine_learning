#!/usr/bin/env python3
"""Inception Network"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    # Initialize kernel weights using He normal initializer
    init = K.initializers.HeNormal()

    # Define input layer with shape (224, 224, 3)
    inputs = K.Input((224, 224, 3))

    # First convolutional block
    # 64 filters, 7x7 kernel size, 2 stride, same padding
    c1 = K.layers.Conv2D(
        64,
        7,
        2,
        activation='relu',
        padding='same',
        kernel_initializer=init)(inputs)

    # Max pooling layer
    # 3x3 pool size, 2 stride, same padding
    Mpool1 = K.layers.MaxPool2D((3, 3), 2, padding='same')(c1)

    # Second convolutional block
    # 192 filters, 3x3 kernel size, 1 stride, same padding
    c2 = K.layers.Conv2D(
        192,
        3,
        1,
        activation='relu',
        padding='same',
        kernel_initializer=init)(Mpool1)

    # Max pooling layer 2
    # 3x3 pool size, 2 stride, same padding
    M2pool = K.layers.MaxPool2D((3, 3), 2, padding='same')(c2)

    # Inception blocks
    inception3a = inception_block(M2pool, [64, 96, 128, 16, 32, 32])
    inception3b = inception_block(inception3a, [128, 128, 192, 32, 96, 64])
    pool3 = K.layers.MaxPooling2D(
        pool_size=(
            3, 3), strides=(
            2, 2), padding="same")(inception3b)
    inception4a = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    inception4b = inception_block(inception4a, [160, 112, 224, 24, 64, 64])
    inception4c = inception_block(inception4b, [128, 128, 256, 24, 64, 64])
    inception4d = inception_block(inception4c, [112, 144, 288, 32, 64, 64])
    inception4e = inception_block(inception4d, [256, 160, 320, 32, 128, 128])
    pool4 = K.layers.MaxPooling2D(
        pool_size=(
            3, 3), strides=(
            2, 2), padding="same")(inception4e)
    inception5a = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    inception5b = inception_block(inception5a, [384, 192, 384, 48, 128, 128])

    # Average pooling layer
    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7), padding="valid")(inception5b)

    # Dropout layer
    dropout = K.layers.Dropout(rate=0.4)(avg_pool)

    # Fully connected layer
    fc = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=init)(dropout)

    # Define the model with input and output layers
    model = K.models.Model(inputs=inputs, outputs=fc)

    return model
