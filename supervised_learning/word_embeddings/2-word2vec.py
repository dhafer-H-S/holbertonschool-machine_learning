#!/usr/bin/env python3
"""Word2Vec Model Training"""

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
    Create and train a gensim Word2Vec model.

    Args:
        sentences (list): List of tokenized sentences to train on.
        size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum occurrences of a word for use in training.
        window (int): Maximum distance between the current and predicted word in a sentence.
        negative (int): Size of negative sampling.
        cbow (bool): If True, use CBOW; if False, use Skip-gram.
        iterations (int): Number of iterations (epochs) to train the model.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads for training.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    sg = 0 if cbow else 1  # CBOW (sg=0) or Skip-gram (sg=1)

    # Initialize the Word2Vec model
    model = gensim.models.Word2Vec(
        sentences=sentences,
        size=size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        negative=negative,
        seed=seed
    )

    # Train the model
    model.train(
        sentences=sentences,
        total_examples=model.corpus_count,
        epochs=iterations
    )

    return model
