# /usr/bin/env python3
"""
extract word 2 vec
convert gensim word2vec model to keras embedding layer

This module provides a function to convert a gensim word2vec model to a Keras embedding layer.
"""

import numpy as np
from keras.layers import Embedding
import gensim
model = __import__('2-word2vec').word2vec_model


def gensim_to_keras(model):
    """Converts a gensim word2vec model to a Keras embedding layer.

    Args:
        model (gensim.models.Word2Vec): The gensim word2vec model.

    Returns:
        keras.layers.Embedding: The Keras embedding layer.

    """
    weights = model.wv.vectors
    vocab_size, embedding_dim = weights.shape
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[weights])
    return embedding_layer
