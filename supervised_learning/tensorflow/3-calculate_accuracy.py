#!/usr/bin/env python3

"""Calculate the accuracy of predictions."""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
        y A placeholder for the labels of the input data.
        y_pred A tensor containing the network's predictions.
    """
    """Compare predicted labels with true labels and get a boolean tensor"""
    correct_predictions = tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1))

    """
    Convert boolean tensor to floating-point values and
    calculate mean accuracy
    """
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    """A tensor containing the decimal accuracy of the prediction."""
    return accuracy
