#!/usr/bin/env python3
"""
RNN Decoder
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention

class RNNDecoder(tf.keras.layers.Layer):
    """RNNDecoder class"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor
        :param vocab: an integer representing the size of
        the output vocabulary
        :param embedding: an integer representing the dimensionality
        of the embedding vector
        :param units: an integer representing the number of hidden
        units in the RNN cell
        :param batch: an integer representing the batch size
        """
        super(RNNDecoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, return_sequences=True,
            return_state=True, recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)
        self.attention = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """
        :param x: a tensor of shape (batch, 1) containing the previous word in
        the target sequence as an index of the target vocabulary
        :param s_prev: a tensor of shape (batch, units)containing the previous
        decoder hidden state
        :param hidden_states: a tensor of shape (batch, input_seq_len, units)
        containing the outputs of the encoder
        :return: y, s
        """
        context, _ = self.attention(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)
        y, s = self.gru(x, initial_state=s_prev)
        y = tf.reshape(y, (-1, y.shape[2]))
        y = self.F(y)
        return y, s
