#!/usr/bin/env python3

import tensorflow as tf


class RNNEncoder:
    """
    A class representing a Recurrent Neural Network Encoder.

    Attributes:
        vocab (int): The size of the input vocabulary.
        embedding (int): The dimensionality of the embedding vector.
        units (int): The number of hidden units in the RNN cell.
        batch (int): The batch size.

    Methods:
        __init__(self, vocab, embedding, units, batch): Initializes the RNNEncoder object.
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
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        Initializes the hidden state to zeros.

        Returns:
            tf.Tensor: The initialized hidden state.
        """
        return tf.zeros([self.batch, self.units])

    def call(self, x, initial):
        """
        Extracts input sequences.

        Args:
            x (tf.Tensor): The input tensor.
            initial (tf.Tensor): The initial hidden state.

        Returns:
            tuple: A tuple containing the outputs and the hidden state.
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
