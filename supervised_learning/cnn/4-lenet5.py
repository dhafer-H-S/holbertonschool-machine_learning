#!/usr/bin/env python3
"""
a function that builds a modified version of LeNet 5 architecture
using tensorflow
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    m = x.shape
    """
    m is the number of images
    """
    x = tf.placeholder(tf.float32, shape=[m, 28, 28, 1])
    """
    place holder containing the inpute images for the network
    """
    y = tf.placeholder(tf.float32, shape=[m, 10])
    """
    place holder containing the one hot labels for the network
    """
    model = tf.keras.Sequential()
    """
    define sequential model
    """
    conv1 = tf.keras.layers.Conv2D(
        fliters=6,
        kernel_size=(
            5,
            5),
        padding='same',
        activation='tanh',
        input_shape=x,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        name='conv1')
    """
    convolution layer 1ers
    """
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))
    """
    max pooling layer 1
    """
    conv2 = tf.keras.layers.Conv2D(
        fliters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='tanh',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        name='conv2')
    """
    convolution layer 2
    """
    pool2 = tf.keras.layers.MaxPooling2D(
        pool_size=(
            2, 2), strides=(
            2, 2), name='pool2')
    """
    max pooling layer 2
    """
    fully_con = tf.keras.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        name='fully_con')
    """
    fully conected layer with 120 nodes
    """
    fully_con2 = tf.keras.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        name='fully_con2')
    """
    fully conected layer with 84 nodes
    """
    fully_con3 = tf.keras.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        name='fully_con3')
    """
    fully conected layer with 10 nodes and softmax activation
    """
    output = tf.nn.softmax(fully_con3)
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=fully_con3)
    """
    we calculate the loss using softmax cross entropy with logits v2"""
    optimizer = tf.train.AdamOptimizer()
    """ we optimize using adam optimizer"""
    train_op = optimizer.minimize(loss)
    """ we train using the optimizer"""
    correct_prediction = tf.equal(tf.argmax(fully_con3, 1), tf.argmax(y, 1))
    """ we check if the prediction is correct"""
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    """ we calculate the accuracy"""
    return output, train_op, loss, accuracy
