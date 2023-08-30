#!/usr/bin/env python3

"""train function that builds, trains, and saves a neural network classifier"""
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        layer_sizes,
        activations,
        alpha,
        iterations,
        save_path="/tmp/model.ckpt"):
    """Build, train, and save a neural network classifier.

    Args:
        X_train: Numpy array containing training input data.
        Y_train: Numpy array containing training labels.
        X_valid: Numpy array containing validation input data.
        Y_valid: Numpy array containing validation labels.
        layer_sizes: List of integers with number of nodes in each layer.
        activations: List of activation functions for each layer.
        alpha: Learning rate.
        iterations: Number of iterations to train over.
        save_path: Path to save the model.

    Returns:
        The path where the model was saved.
    """
    tf.reset_default_graph()

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(iterations + 1):
            if i == 0 or i == iterations or i % 100 == 0:
                train_cost = sess.run(loss, feed_dict={x: X_train, y: Y_train})
                train_acc = sess.run(
                    accuracy, feed_dict={
                        x: X_train, y: Y_train})
                valid_cost = sess.run(loss, feed_dict={x: X_valid, y: Y_valid})
                valid_acc = sess.run(
                    accuracy, feed_dict={
                        x: X_valid, y: Y_valid})

                print(f"After {i} iterations:")
                print(f"\tTraining Cost: {train_cost}")
                print(f"\tTraining Accuracy: {train_acc}")
                print(f"\tValidation Cost: {valid_cost}")
                print(f"\tValidation Accuracy: {valid_acc}")

            if i < iterations:
                sess.run(train_op, feed_dict={x: X_train, y: Y_train})

        saver = tf.train.Saver()
        save_path = saver.save(sess, save_path)

    return save_path
