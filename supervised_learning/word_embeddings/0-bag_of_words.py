#!/usr/bin/env python3
"""woord of bag"""

import numpy as np

def bag_of_words(sentences, vocab=None):
    if vocab is None:
        features = sorted(set(word for sentence in sentences for word in sentence.split()))
    else:
        features = sorted(vocab)
    
    embeddings = np.zeros((len(sentences), len(features)))
    
    for i, sentence in enumerate(sentences):
        for j, word in enumerate(features):
            embeddings[i, j] = sentence.split().count(word)
    
    return embeddings, features