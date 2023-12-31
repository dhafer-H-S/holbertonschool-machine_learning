#!/usr/bin/env python3
""" calculate cost function with L2 regularization """
import tensorflow.compat.v1 as tf


def l2_reg_cost(cost):
    """
    cost is a tensor containg the cost of the network without
    L2 regularization
    """
    l2_reg = tf.losses.get_regularization_losses()
    return cost + l2_reg
