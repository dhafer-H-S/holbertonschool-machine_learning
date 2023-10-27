#!/usr/bin/env python3
"""Inception network."""
import tensorflow.keras as K  # Import the Keras module as K
inception_block = __import__('0-inception_block').inception_block  # Import the inception_block function from the 0-inception_block module


def inception_network():  # Define the inception_network function
    """Inception network."""  # Function docstring
    init = K.initializers.HeNormal()  # Initialize the weights using the He normal initializer
    lin = K.Input((224, 224, 3))  # Define the input layer with shape (224, 224, 3)
    C1 = K.layers.Conv2D(64, 7, 2, activation='relu',  # Add a convolutional layer with 64 filters, a 7x7 kernel size, and a stride of 2, followed by max pooling with a pool size of 3x3 and a stride of 2
                         padding='same', kernel_initializer=init)(lin)
    m1 = K.layers.MaxPool2D((3, 3), 2, padding='same')(C1)  # Add max pooling with a pool size of 3x3 and a stride of 2
    C2 = K.layers.Conv2D(192, 3, 1, activation='relu',  # Add a second convolutional layer with 192 filters, a 3x3 kernel size, and a stride of 1, followed by max pooling with a pool size of 3x3 and a stride of 2
                         padding='same', kernel_initializer=init)(m1)
    m2 = K.layers.MaxPool2D((3, 3), 2, padding='same')(C2)  # Add max pooling with a pool size of 3x3 and a stride of 2
    b1 = inception_block(m2, [64, 96, 128, 16, 32, 32])  # Add the first inception block with the specified filter sizes
    b2 = inception_block(b1, [128, 128, 192, 32, 96, 64])  # Add the second inception block with the specified filter sizes
    m3 = K.layers.MaxPool2D((3, 3), 2, padding='same')(b2)  # Add max pooling with a pool size of 3x3 and a stride of 2
    b3 = inception_block(m3, [192, 96, 208, 16, 48, 64])  # Add the third inception block with the specified filter sizes
    b4 = inception_block(b3, [160, 112, 224, 24, 64, 64])  # Add the fourth inception block with the specified filter sizes
    b5 = inception_block(b4, [128, 128, 256, 24, 64, 64])  # Add the fifth inception block with the specified filter sizes
    b6 = inception_block(b5, [112, 144, 288, 32, 64, 64])  # Add the sixth inception block with the specified filter sizes
    b7 = inception_block(b6, [256, 160, 320, 32, 128, 128])  # Add the seventh inception block with the specified filter sizes
    m4 = K.layers.MaxPooling2D((3, 3), 2, padding='same')(b7)  # Add max pooling with a pool size of 3x3 and a stride of 2
    b8 = inception_block(m4, [256, 160, 320, 32, 128, 128])  # Add the eighth inception block with the specified filter sizes
    b9 = inception_block(b8, [384, 192, 384, 48, 128, 128])  # Add the ninth inception block with the specified filter sizes
    avg1 = K.layers.AveragePooling2D((7, 7), 1)(b9)  # Add average pooling with a pool size of 7x7
    d1 = K.layers.Dropout(0.4)(avg1)  # Add a dropout layer with a rate of 0.4
    softmax = K.layers.Dense(1000, activation='softmax')(d1)  # Add a fully connected layer with 1000 units and a softmax activation function
    model = K.models.Model(lin, softmax)  # Create the Keras model with the input and output layers
    return model  # Return the model
