#!/usr/bin/env python3
"""attention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """calculate attention for machine translation"""

    def __init__(self, units):
        """constructor
        units is an integer representing the number of hidden units in the
        alignment model"""
        super(SelfAttention, self).__init__()
        # W is hidden state of encoder
        self.W = tf.keras.layers.Dense(units)
        # U is hidden state of decoder
        self.U = tf.keras.layers.Dense(units)
        # V is the tanh of the sum of the outputs of W and U
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """calls the attention
        s_prev is a tensor of shape (batch, units) containing the previous
        decoder hidden state
        hidden_states is a tensor of shape (batch, input_seq_len, units)
        containing the outputs of the encoder
        Returns: context, weights"""
        s_prev = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)
        return context, weights
