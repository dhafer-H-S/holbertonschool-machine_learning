#!/usr/bin/env python3
"""
transformer network
"""
import tensorflow as tf


class Transformer(tf.keras.Model):
    """Transformer model for machine translation."""

    def __init__(self, N, dm, h, hidden, target_vocab, input_vocab, max_len):
        super(Transformer, self).__init__()

        """Initialize encoder and decoder"""
        self.encoder = Encoder(N, dm, h, hidden, input_vocab, max_len)
        self.decoder = Decoder(N, dm, h, hidden, target_vocab, max_len)

    def call(self, inputs, target, training, encoder_mask, combined_mask):
        """
        Forward pass through the model.
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output,
                                      training, combined_mask)
        return decoder_output

