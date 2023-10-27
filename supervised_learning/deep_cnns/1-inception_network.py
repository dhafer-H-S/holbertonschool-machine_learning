#!/usr/bin/env python3
"""function that builds inception block in a modified way"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """function that builds inception block in a modified way"""
    init = K.initializers.HeNormal()
    inputs = K.Input((224, 224, 3))
    """ 3 channels with 224*224 pixels, RGB"""
    """first convolution block"""
    c1 = K.layers.Conv2D(
        64,
        7,
        2,
        activation='relu',
        padding='same',
        kernel_initializer=init)(inputs)
    """64 filters, 7*7 kernel, 2 stride, same padding"""
    """max pooling layer"""
    Mpool1 = K.layers.MaxPool2D((3, 3), 2, padding='same')(c1)
    """second convolution block"""
    c2 = K.layers.Conv2D(
        192,
        3,
        1,
        activation='relu',
        padding='same',
        kernel_initializer=init)(Mpool1)
    """192 filters, 3*3 kernel, 1 stride, same padding"""
    M2pool = K.layers.MaxPool2D((3, 3), 2, padding='same')(c2)
    """max pooling layer 2"""
    """inception blocks"""
    i1 = inception_block(M2pool, [64, 96, 128, 16, 32, 32])
    i2 = inception_block(i1, [128, 128, 192, 32, 96, 64])
    """max pooling layer 3"""
    M3pool = K.layers.MaxPool2D((3, 3), 2, padding='same')(i2)
    """inception block 3"""
    i4 = inception_block(M3pool, [192, 96, 208, 16, 48, 64])
    """inception block 4"""
    i5 = inception_block(i4, [160, 112, 224, 24, 64, 64])
    """inception block 5"""
    i6 = inception_block(i5, [128, 128, 256, 24, 64, 64])
    """inception block 6"""
    i7 = inception_block(i6, [112, 144, 288, 32, 64, 64])
    """inception block 7"""
    """max pooling layer 4"""
    M4pool = K.layers.MaxPool2D((3, 3), 2, padding='same')(i8)
    i8 = inception_block(i7, [256, 160, 320, 32, 128, 128])
    """inception block 8"""
    i9 = inception_block(M4pool, [256, 160, 320, 32, 128, 128])
    """inception block 9"""
    """average pooling layer"""
    avg_pool = K.layers.AveragePooling2D((7, 7), strides=1)(i9)
    """dropout layer"""
    dropout = K.layers.Dropout(0.4)(avg_pool)
    """softmax layer"""
    linear = K.layers.Dense(
        1000,
        activation='softmax',
        kernel_initializer=init)(dropout)
    """model"""
    model = K.Model(inputs=inputs, outputs=linear)
    return model
