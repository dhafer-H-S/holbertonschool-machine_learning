#!/usr/bin/env python3
""" RMSProp optimzation with tenserflow function """
import tensorflow.compat.v1 as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    optimizer = tf.train.RMSPropOptimizer(learning_rate=alpha, decay=beta2, epsilon=epsilon)
    return optimizer.minimze(loss)