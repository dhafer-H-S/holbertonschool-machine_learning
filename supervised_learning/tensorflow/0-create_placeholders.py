#!/usr/bin/env python3


import tensorflow.compat.v1 as tf


""" a place holder function return x, and y for the neural network"""
def create_placeholders(nx, classes):

    """nx the number of feature colims in our data"""
    """number of classes in our classifier"""
    """ x place holder for the inpute data"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    """ y place holder for the one hot labels for the inpute data"""
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    """ return place holders for the data"""
    return x, y
