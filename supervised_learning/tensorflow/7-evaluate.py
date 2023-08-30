#!/usr/bin/env python3

import tensorflow.compat.v1 as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop

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

    x, y = create_placeholders(X.shape[1], Y.shape[1])
    y_pred = forward_prop(x, [100], [tf.nn.relu])
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, save_path)
        prediction = sess.run(y_pred, feed_dict={x: X, y: Y})
        acc = sess.run(accuracy, feed_dict={x: X, y: Y})
        loss_value = sess.run(loss, feed_dict={x: X, y: Y})

    return prediction, acc, loss_value
