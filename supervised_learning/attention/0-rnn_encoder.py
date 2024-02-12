#!/usr/bin/env python3
"""RNN ECODER """

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    A class representing a Recurrent Neural Network Encoder.

    Attributes:
        vocab (int): The size of the input vocabulary.
        embedding (int): The dimensionality of the embedding vector.
        units (int): The number of hidden units in the RNN cell.
        batch (int): The batch size.

    Methods:
        __init__(self, vocab, embedding, units, batch): Initializes
        the RNNEncoder object.
        initialize_hidden_state(self): Initializes the hidden state to zeros.
        call(self, x, initial): Extracts input sequences.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Initializes the RNNEncoder object.

        Args:
            vocab (int): The size of the input vocabulary.
            embedding (int): The dimensionality of the embedding vector.
            units (int): The number of hidden units in the RNN cell.
            batch (int): The batch size.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True,
            )

    def initialize_hidden_state(self):
        """
        Initializes the hidden state to zeros.

        Returns:
            tf.Tensor: The initialized hidden state.
        """
        return tf.zeros([self.batch, self.units])

    def call(self, x, initial):
        """
        Perform the forward pass of the RNN Encoder.
        Args:
            x (tf.Tensor): Input tensor of shape
            (batch_size, sequence_length, input_dim).
            initial (tf.Tensor): Initial state tensor of shape
            (batch_size, units).
        Returns:
            outputs (tf.Tensor): Output tensor of shape
            (batch_size, sequence_length, units).
            state (tf.Tensor): Final state tensor of shape
            (batch_size, units).
        """
        x = self.embedding
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
