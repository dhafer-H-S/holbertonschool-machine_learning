#!/usr/bin/env python3
"""fasttext model"""

import numpy as np
from gensim.models import FastText
from gensim.test.utils import common_texts


def fasttext_model(
        sentences,
        size=100,
        min_count=5,
        negative=5,
        window=5,
        cbow=True,
        iterations=5,
        seed=0,
        workers=1):
    """fasttest model"""
    sg = 0 if cbow else 1
    model = FastText(
        vector_size=size,
        window=window,
        min_count=min_count,
        sg=sg,
        seed=seed,
        negative=negative,
        workers=workers)
    model.build_vocab(corpus_iterable=sentences)
    model.train(
        corpus_iterable=sentences,
        total_examples=len(sentences),
        epochs=iterations)
    return model
