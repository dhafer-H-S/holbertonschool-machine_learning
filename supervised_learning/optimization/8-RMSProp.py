#!/usr/bin/env python3
""" RMSProp optimzation with tenserflow function """


import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    create the training operation for a neural network in tenserflow using
    RMSProp optimization algorithm
    """
    """ alpha learning rate """
    """ beta2 the RMSProp weight """
    """ epilson is a small  number to avid divisin by zero """
    """ loss of the model network """
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha,
                                          decay=beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
