#!/usr/bin/env python3
"""attention"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """calculate attention for machine translation"""

    def __init__(self, units):
        """
        Initializes a SelfAttention object.

        Parameters:
        - units: An integer representing the number of hidden units
        in the alignment model.
        W: A Dense layer that processes the previous decoder hidden state.
        U: A Dense layer that processes the encoder hidden states.
        V: A Dense layer that processes the combined output of W and U
        to produce the attention scores.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """calls the attention
        s_prev is a tensor of shape (batch, units) containing the previous
        decoder hidden state
        hidden_states is a tensor of shape (batch, input_seq_len, units)
        containing the outputs of the encoder
        Returns: context, weights"""
    def call(self, s_prev, hidden_states):
        s_prev_expanded = tf.expand_dims(s_prev, 1)
        score = self.V(tf.nn.tanh(self.W(s_prev_expanded) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
