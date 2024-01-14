#!/usr/bin/env python3
"""woord of bag"""

import numpy as np
import re


def bag_of_words(sentences, vocab=None):
    """Creates a bag of words embedding matrix"""
    def clean_sentence(sentence):
        """Clean sentence before embedding"""
        return re.sub(r"\b\w{1}\b", "", re.sub(
            r"[^a-zA-Z0-9\s]", " ", sentence.lower())).split()

    if vocab is None:
        vocab = sorted(
            set(
                word for sentence in sentences for word
                in clean_sentence(sentence)))

    embeddings = np.zeros((len(sentences), len(vocab)))

    for i, sentence in enumerate(sentences):
        cleaned_sentence = clean_sentence(sentence)
        for word in cleaned_sentence:
            if word in vocab:
                embeddings[i, vocab.index(word)] += 1

    return embeddings.astype(int), vocab
