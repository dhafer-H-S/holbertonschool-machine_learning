#!/usr/bin/env python3
"""function that builds inception block in a modified way"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    input_shape = (224, 224, 3)
    inputs = K.layers.Input(shape=input_shape)

    # First convolutional layer with 64 filters and a 7x7 kernel size,
    # followed by max pooling with a pool size of 3x3 and a stride of 2
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=7,
        strides=2,
        padding='same',
        activation='relu')(inputs)
    pool1 = K.layers.MaxPooling2D(
        pool_size=3, strides=2, padding='same')(conv1)

    # Second convolutional layer with 64 filters and a 1x1 kernel size,
    # followed by a third convolutional layer with 192 filters and a 3x3
    # kernel size, and max pooling with a pool size of 3x3 and a stride of 2
    conv2r = K.layers.Conv2D(
        filters=64,
        kernel_size=1,
        padding='same',
        activation='relu')(pool1)
    conv2 = K.layers.Conv2D(
        filters=192,
        kernel_size=3,
        padding='same',
        activation='relu')(conv2r)
    pool2 = K.layers.MaxPooling2D(
        pool_size=3, strides=2, padding='same')(conv2)

    # Two inception blocks with 64, 96, 128, 16, 32, 32 and 128, 128, 192, 32,
    # 96, 64 filters, respectively
    inception1 = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    inception2 = inception_block(inception1, [128, 128, 192, 32, 96, 64])

    # Max pooling with a pool size of 3x3 and a stride of 2, followed by two
    # inception blocks with 192, 96, 208, 16, 48, 64 and 160, 112, 224, 24,
    # 64, 64 filters, respectively
    pool3 = K.layers.MaxPooling2D(
        pool_size=3,
        strides=2,
        padding='same')(inception2)
    inception3 = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    inception4 = inception_block(inception3, [160, 112, 224, 24, 64, 64])

    # Average pooling with a pool size of 7x7, followed by a dropout layer
    # with a rate of 0.4, and a fully connected layer with 1000 units and a
    # softmax activation function
    avg_pool = K.layers.GlobalAveragePooling2D()(inception4)
    dropout = K.layers.Dropout(rate=0.4)(avg_pool)
    outputs = K.layers.Dense(units=1000, activation='softmax')(dropout)

    # Create the Keras model with the input and output layers
    model = K.models.Model(inputs=inputs, outputs=outputs)

    return model
