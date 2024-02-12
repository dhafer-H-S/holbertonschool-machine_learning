#!/usr/bin/env python3
"""attention"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """rnn encoder"""

    def __init__(self, vocab, embedding, units, batch):
        """constructor
        embedding is an int representing the dimensionality of the embedding
        vector
        gru recurrent layer to encode the input sequence"""
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab,
                                                   embedding)
        self.gru = tf.keras.layers.GRU(units,
                                       recurrent_initializer='glorot_uniform',
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """initializes hidden state"""
        return tf.zeros([self.batch, self.units])

    def call(self, x, initial):
        """calls encoder
        use embedding layer to create an embedding vector
        then pass the embedding vector to the GRU layer
        Returns: outputs, hidden"""
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden