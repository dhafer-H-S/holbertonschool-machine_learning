#!/usr/bin/env python3
"""
a function that builds a modified version of LeNet 5 architecture
using tensorflow
"""
import tensorflow.compat.v1 as tf


def accuracy(y, y_pred):
    """Accuracy of prediction."""
    y = tf.argmax(y, axis=1)
    y_pred = tf.argmax(y_pred, axis=1)
    correct = tf.equal(y, y_pred)
    return tf.reduce_mean(tf.cast(correct, dtype='float'))


def lenet5(x, y):
    """
    function that builds a modified version of LeNet 5 architecture
    """
    m = x.shape
    """
    m is the number of images
    """
    conv1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(
            5,
            5),
        padding='same',
        activation='tanh',
        input_shape=x,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        name='conv1')(x)
    """
    convolution layer 1
    """
    pool1 = tf.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv1)
    """
    max pooling layer 1
    """
    conv2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='tanh',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        name='conv2')(pool1)
    """
    convolution layer 2
    """
    pool2 = tf.layers.MaxPooling2D(
        pool_size=(
            2, 2), strides=(
            2, 2), name='pool2')(conv2)
    """
    max pooling layer 2
    """
    flatten = tf.layers.Flatten()(pool2)
    fully_con = tf.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        name='fully_con')(flatten)
    """
    fully connected layer with 120 nodes
    """
    fully_con2 = tf.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        name='fully_con2')(fully_con)
    """
    fully connected layer with 84 nodes
    """
    fully_con3 = tf.layers.Dense(
        units=10,
        kernel_initializer=tf.keras.initializers.VarianceScaling(scale=2.0),
        name='fully_con3')(fully_con2)
    """
    fully connected layer with 10 nodes and no activation function
    """
    y_pred = tf.nn.softmax(fully_con3)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=fully_con3)
    train_op = tf.train.AdamOptimizer().minimize(loss)
    accuracy = accuracy(y, y_pred)

    return y_pred, train_op, loss, accuracy
