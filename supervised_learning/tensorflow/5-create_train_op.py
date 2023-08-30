#!/usr/bin/env python3

"""
a fucntion that creats a training operationn for the network
using gradient descent
"""
import tensorflow.compat.v1 as tf


def create_train_op(loss, alpha):
    """ loss is the loss of the networks prediction"""
    """alpha is the learning rate"""

    """ an operation that trains the network usnig gradent descent """
    """This line creates an instance of the GradientDescentOptimizer 
    from TensorFlow's training module"""
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    """ this function call generates the operation that minimizes
    the provided loss using the optimization methode defined by the optimizer"""
    """ it calculates gradient with respect to the model's trainble variables
    and updates them based on the learning rate"""
    train_op = optimizer.minimize(loss)
    
    return train_op
