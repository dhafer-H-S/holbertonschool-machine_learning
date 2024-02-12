#!/usr/bin/env python3
"""self attention"""

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """a self attention class"""

    def __init__(self, units):
        super(SelfAttention, self).__init__()
        """
        units is a number representing the number of hidden units in the
        alignement model
        """
        self.W = tf.keras.layers.Dense(units=units, activation='relu')
        """decoder hidden state transformation"""
        self.U = tf.keras.layers.Dense(units=units, activation='relu')
        """encoder hidden states transformation"""
        self.V = tf.keras.layers.Dense(units=1)
        """attention scoring"""

    def call(self, s_prev, hidden_states):
        """
        call function to calcualte attention weights
        s_prev is a tensor of shape (batch, units) containing the previous
        decoder hidden state
        hidden_states is a tensor of shape (batch, inut_seq_len, units)
        conatining the output of the encoder
        """
        score = self.V(tf.nn.tanh(self.W(tf.expand_dims(s_prev, 1)) + self.U(hidden_states)))
        attention_weights = tf.nn.softmax(score, axis=1)
        attention_weights = tf.expand_dims(attention_weights, -1)  # Expand the dimensions of attention_weights
        context_vector = attention_weights * hidden_states
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
