#!/usr/bin/env python3

""" function to calculate the loss """
import tensorflow.compat.v1 as tf


def calculate_loss(y, y_pred):
    """
    y is a aplace holder for the labels of the inpute data
    y_pred a tensor containing the network's predictions
    """

    """tf.reduce. mean calculate the mean of the computed cross entropy losses"""
    """this function calculates th esoftmax cross entropy loss between the
    true lbels yan the netwwork prediction y_pred it computes the cross entropy
    loss for each example and classs and the takes the mean over all examples """
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=y, logits=y_pred))
    """ a tensor containing the loss of the prediction"""
    return loss
