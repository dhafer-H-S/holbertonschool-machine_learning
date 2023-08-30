#!/usr/bin/env python3

"""Evaluate the output of a neural network."""
import tensorflow.compat.v1 as tf


def evaluate(X, Y, save_path):
    """
        X Numpy array containing input data to evaluate.
        Y Numpy array containing one-hot labels for X.
        save_path Location to load the model from.
    """
    """ Reset the TensorFlow graph to ensure a clean slate """
    tf.reset_default_graph()

    with tf.Session() as sess:
        """ Import the saved model's graph """
        saver = tf.train.import_meta_graph(save_path + '.meta')
        """ Restore the model's trained parameters """
        saver.restore(sess, save_path)
        """Retrieve tensors from the collections using their names"""
        """Input placeholder tensor"""
        x = tf.get_collection('x')[0]
        """Label placeholder tensor"""
        y = tf.get_collection('y')[0]
        """Prediction tensor"""
        y_pred = tf.get_collection('y_pred')[0]
        """Accuracy tensor"""
        accuracy = tf.get_collection('accuracy')[0]
        """Loss tensor"""
        loss = tf.get_collection('loss')[0]
        """Create a feed dictionary for input and labels"""
        feed_dict = {x: X, y: Y}
        """Run the session to compute prediction, accuracy, and loss values"""
        y_pred_val, accuracy_val, loss_val = sess.run(
            [y_pred, accuracy, loss], feed_dict=feed_dict)
    """Return the computed values"""
    return y_pred_val, accuracy_val, loss_val
