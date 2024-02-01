#!/usr/bin/env python3
"""N-gram BLUE score"""


from collections import Counter
import numpy as np


def ngram_bleu(references, sentence, n):
    """
    refrences is the list of refrences to translate
    sentence is a list containing the model proposed sentence
    n is the number of n gram to use for evoluation
    """
    """Create n-grams for sentence"""
    sentence_ngrams = Counter([tuple(sentence[i:i + n])
                              for i in range(len(sentence) - n + 1)])

    max_counts = {}
    for reference in references:
        """Create n-grams for reference"""
        reference_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) - n + 1)])
        for ngram in sentence_ngrams:
            max_counts[ngram] = max(max_counts.get(
                ngram, 0), reference_ngrams[ngram])

    clipped_counts = {
        ngram: min(
            count,
            max_counts.get(
                ngram,
                0)) for ngram,
        count in sentence_ngrams.items()}

    bleu_score = sum(clipped_counts.values()) / max(len(sentence_ngrams), 1)

    """Calculate brevity penalty"""
    closest_ref_len = min((abs(len(sentence) - len(ref)), len(ref))
                          for ref in references)[1]
    if len(sentence) > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / len(sentence))

    return brevity_penalty * bleu_score
