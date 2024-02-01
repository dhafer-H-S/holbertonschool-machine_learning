#!/usr/bin/env python3
"""unigram BLEU score"""

from collections import Counter
import numpy as np

def uni_bleu(references, sentence):
    """
    references is a list of reference translations
    each reference translation is a list of the words in the translation
    sentence is a list containing the model proposed sentence
    Returns: the unigram BLEU score
    """
    sentence_counter = Counter(sentence)
    """count if the word exits moore then one """
    max_counts = {}
    """empty list"""
    for reference in references:
        reference_counter = Counter(reference)
        for word in sentence_counter:
            max_counts[word] = max(max_counts.get(word, 0), reference_counter[word])
    clipped_counts = {word: min(count, max_counts.get(word, 0)) for word, count in sentence_counter.items()}
    blue_score = sum(clipped_counts.values()) / max(len(sentence), 1)
    """calculate brevity penalty"""
    closest_ref_len = min((abs(len(sentence) - len(ref)), len(ref)) for ref in references)[1]
    if len(sentence) > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = np.exp(1 - closest_ref_len / len(sentence))
    score = brevity_penalty * blue_score
    return score