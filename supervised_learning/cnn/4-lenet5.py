#!/usr/bin/env python3
"""
a function that builds a modified version of LeNet 5 architecture
using tensorflow
"""
import tensorflow.compat.v1 as tf


def lenet5(x, y):
    """
    a modified version of the LeNet-5 architecture using tensorflow
    """
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
    model.add(
        tf.keras.layers.Conv2D(
            fliters=6,
            kernel_size=(
                5,
                5),
            padding='same',
            activation='tanh',
            input_shape=x,
            kernel_initializer=tf.keras.initializers.VarianceScaling(
                scale=2.0)))
    """
    convolution layer 1ers
    """
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    """
    max pooling layer 1
    """
    model.add(
        tf.keras.layers.Conv2D(
            fliters=16,
            kernel_size=(
                5,
                5),
            padding='valid',
            activation='tanh'),
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0))
    """
    convolution layer 2
    """
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    """
    max pooling layer 2
    """
    model.add(
        tf.keras.layers.Dense(
            units=120,
            activation='relu'),
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0))
    """
    fully conected layer with 120 nodes
    """
    model.add(
        tf.keras.layers.Dense(
            units=84,
            activation='relu'),
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0))
    """
    fully conected layer with 84 nodes
    """
    model.add(
        tf.keras.layers.Dense(
            units=10,
            activation='softmax'),
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=2.0))
    """
    fully conected layer with 10 nodes and softmax activation
    """
    logits = model(x)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=y, logits=logits))
    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return logits, train_op, loss, accuracy
