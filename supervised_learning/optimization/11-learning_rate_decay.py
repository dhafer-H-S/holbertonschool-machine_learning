#!/usr/bin/env python3
""" update the learning rate decay using inverse time decay """


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    alpha is the original learning rate
    decay_rate is the weight used to determine the rate at which alpha will decay
    global_step is the number of passes of gradient descent that have elapsed
    decay_step is the number of passes of gradient descent that should occur
    before alpha is decayed further
    the learning rate decay should occur in a stepwise fashion
    Returns: the updated value for alpha
    """
    num_decay_step = global_step // decay_step
    alpha = alpha / (1 + decay_rate * num_decay_step)
    return alpha
