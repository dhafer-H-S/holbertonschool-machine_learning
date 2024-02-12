#!/usr/bin/env python3

import tensorflow as tf


class RNNEncoder:
    """creat a class recurent neural network"""
    def __init__(self, vocab, embedding, units, batch):
        """
        vocab is an integer representing the size of the inpute vocabulary
        embedding is an integer representing the dimensionality of the embedding vector
        units is an integer representing the number of hidden units in the rnn cell
        batch is an integer representing the batch size
        """
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    def initialize_hidden_state(self):
        """
        fuction to initialize the the hiden state to zeros
        """
        return tf.zeros([self.batch, self.units])

    def call(self, x, initial):
        """
        function to extract inpute sequences
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden