#!/usr/bin/env python3
"""tf-idf"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """TF-IDF"""
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    X = vectorizer.fit_transform(sentences)
    return X.toarray(), vectorizer.get_feature_names_out()
