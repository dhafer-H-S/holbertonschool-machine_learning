#!/usr/bin/env python3

import tensorflow.compat.v1 as tf


""" function to updzte the parameters in the neural network and
evaluet the process"""
def evaluate(X, Y, save_path):
    """Evaluate the output of a neural network.

    Args:
        X: Numpy array containing input data to evaluate.
        Y: Numpy array containing one-hot labels for X.
        save_path: Location to load the model from.

    Returns:
        The network's prediction, accuracy, and loss.
    """
    tf.reset_default_graph()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(save_path + '.meta')
        saver.restore(sess, save_path)

        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        y_pred = tf.get_collection('y_pred')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]

        feed_dict = {x: X, y: Y}
        y_pred_val, accuracy_val, loss_val = sess.run(
            [y_pred, accuracy, loss], feed_dict=feed_dict)

    return y_pred_val, accuracy_val, loss_val
