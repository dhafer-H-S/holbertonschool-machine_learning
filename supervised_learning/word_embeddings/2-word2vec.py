#!/usr/bin/env python3
"""word2vec"""
"""
Write a function def word2vec_model(sentences, size=100, min_count=5, window=5, negative=5, cbow=True, iterations=5, seed=0, workers=1): that creates and trains a gensim word2vec model:

sentences is a list of sentences to be trained on
size is the dimensionality of the embedding layer
min_count is the minimum number of occurrences of a word for use in training
window is the maximum distance between the current and predicted word within a sentence
negative is the size of negative sampling
cbow is a boolean to determine the training type; True is for CBOW; False is for Skip-gram
iterations is the number of iterations to train over
seed is the seed for the random number generator
workers is the number of worker threads to train the model
Returns: the trained model
"""




import gensim.models
import numpy as np
import pandas as pd
import nltk
import gensim

def word2vec_model(
        sentences,
        size=100,
        min_count=5,
        window=5,
        negative=5,
        cbow=True,
        iterations=5,
        seed=0,
        workers=1):
    """
    Create and train a gensim word2vec model.

    Args:
        sentences (list): List of sentences to be trained on.
        size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum number of occurrences of a word for use in training.
        window (int): Maximum distance between the current and predicted word within a sentence.
        negative (int): Size of negative sampling.
        cbow (bool): Boolean to determine the training type; True is for CBOW, False is for Skip-gram.
        iterations (int): Number of iterations to train over.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads to train the model.

    Returns:
        Trained gensim word2vec model.
    """
    if cbow:
        model = gensim.models.Word2Vec(
            sentences,
            vector_size=size,
            min_count=min_count,
            window=window,
            negative=negative,
            sg=0,
            seed=seed,
            workers=workers )
    else:
        model = gensim.models.Word2Vec(
            sentences,
            vector_size=size,
            min_count=min_count,
            window=window,
            negative=negative,
            sg=1,
            seed=seed,
            workers=workers )
    return model
