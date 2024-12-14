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
        :param vocab: an integer representing the size of the output vocabulary
        :param embedding: an integer representing the dimensionality
        of the embedding vector
        :param units: an integer representing the number of hidden
        units in the RNN cell
        :param batch: an integer representing the batch size
        """
        super(RNNDecoder, self).__init__()
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True)
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        :param x: a tensor of shape (batch, 1) containing the previous word in
        the target sequence as an index of the target vocabulary
        :param s_prev: a tensor of shape (batch, units) containing the previous
        decoder hidden state
        :param hidden_states: a tensor of shape (batch, input_seq_len, units)
        containing the outputs of the encoder
        :return: y, s
        """
        units = s_prev.shape[1]
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        embed = self.embedding(x)
        concat = tf.concat([tf.expand_dims(context, 1), embed], axis=-1)
        outputs, hidden = self.gru(concat)
        outputs = tf.reshape(outputs, (outputs.shape[0], outputs.shape[2]))
        return self.F(outputs), hidden
