#!/usr/bin/env python3
"""
Create The LeNet-5 Convlutional neural network.

The model should consist of the following layers in order:
    Convolutional layer with 6 kernels of shape 5x5 with same padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Convolutional layer with 16 kernels of shape 5x5 with valid padding
    Max pooling layer with kernels of shape 2x2 with 2x2 strides
    Fully connected layer with 120 nodes
    Fully connected layer with 84 nodes
    Fully connected softmax output layer with 10 nodes
"""
import tensorflow.keras as K


def lenet5(X):
    """
    a modified version of the LeNet-5 architecture using keras
    """
      # Define Sequential Model
    model = K.Sequential()
    
    # C1 Convolution Layer
    model.add(K.layers.Conv2D(filters=6, strides=(1,1), kernel_size=(5,5), activation='relu', input_shape=X))
    
    # S2 SubSampling Layer
    model.add(K.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))

    # C3 Convolution Layer
    model.add(K.layers.Conv2D(filters=6, strides=(1,1), kernel_size=(5,5), activation='relu'))

    # S4 SubSampling Layer
    model.add(K.layers.AveragePooling2D(pool_size=(2,2), strides=(2,2)))

    # C5 Fully Connected Layer
    model.add(K.layers.Dense(units=120, activation='relu'))

    # Flatten the output so that we can connect it with the fully connected layers by converting it into a 1D Array
    model.add(K.layers.Flatten())

    # FC6 Fully Connected Layers
    model.add(K.layers.Dense(units=84, activation='relu'))

    # Output Layer
    model.add(K.layers.Dense(units=10, activation='softmax'))

    # Compile the Model
    model.compile(loss='categorical_crossentropy', optimizer=K.AdamOptimizer(), metrics=['accuracy'])

    return model
