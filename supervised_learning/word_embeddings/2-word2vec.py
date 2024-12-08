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
        epochs=5,
        seed=0,
        workers=1):
    """
    Create and train a gensim Word2Vec model.

    Args:
        sentences (list): List of tokenized sentences to train on.
        size (int): Dimensionality of the embedding layer.
        min_count (int): Minimum occurrences of a word for use in training.
        window (int): Maximum distance between the current and
        predicted word in a sentence.
        negative (int): Size of negative sampling.
        cbow Continuous Bag of Words Model(bool): If True, use CBOW
        if False, use Skip-gram.
        iterations (int): Number of iterations (epochs) to train the model.
        seed (int): Seed for the random number generator.
        workers (int): Number of worker threads for training.

    Returns:
        gensim.models.Word2Vec: Trained Word2Vec model.
    """
    sg = 1 if not cbow else 0
    model = gensim.models.Word2Vec(
        sentences,
        vector_size=size,
        min_count=min_count,
        window=window,
        negative=negative,
        sg=sg,
        epochs=epochs,
        seed=seed,
        workers=workers)
    model.build_vocab(sentences)
    model.train(
        sentences,
        total_examples=model.corpus_count,
        epochs=model.epochs)
    return model
