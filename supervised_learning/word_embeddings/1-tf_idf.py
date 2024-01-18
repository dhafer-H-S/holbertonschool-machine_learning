#!/usr/bin/env python3
"""tf-idf"""

import numpy as np
import re


def tf_idf(sentences, vocab=None):
    """tf-idf"""
    def clean_data(sentence):
        return re.sub(r"\b\w{1}\b", "", re.sub(
            r"[^a-zA-Z0-9\s]", " ", sentence.lower())).split()
    if vocab is None:
        vocab = []
        for sentence in sentences:
            for word in clean_data(sentence):
                vocab.append(word)

    embeddings = np.zeros((len(sentences), len(vocab)))
    for i, sentence in enumerate(sentences):
        for word in vocab:
            tf = ((sentence.count(word)) / len(sentence))
            idf = np.log(len(sentences) /
                         (sum(word in s for s in sentences)))
            tf_idf = tf * idf
            embeddings[i, vocab.index(word)] = tf_idf
    return embeddings, vocab
