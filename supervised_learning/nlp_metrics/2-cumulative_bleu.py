#!/usr/bin/env python3
"""N-gram BLUE score"""


from collections import Counter
from math import exp, log


def cumulative_bleu(references, sentence, n):
    """
    Calculate the cumulative BLEU score for a given sentence.

    Args:
        references (list): A list of reference sentences.
        sentence (list): The candidate sentence to evaluate.
        n (int): The maximum n-gram order to consider.

    Returns:
        float: The cumulative BLEU score.

    """
    precisions = []
    for i in range(1, n + 1):
        sentence_ngrams = Counter([tuple(sentence[j:j + i])
                                  for j in range(len(sentence) - i + 1)])
        max_counts = {}
        for reference in references:
            reference_ngrams = Counter(
                [tuple(reference[j:j + i]) for j in range(len(reference) - i + 1)])
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
        precisions.append(sum(clipped_counts.values()) /
                          sum(sentence_ngrams.values()))
    bleu_score = exp(sum(log(p) for p in precisions) / len(precisions))
    closest_ref_len = min((abs(len(sentence) - len(ref)), len(ref))
                          for ref in references)[1]
    if len(sentence) > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = exp(1 - closest_ref_len / len(sentence))
    return round(brevity_penalty * bleu_score, 10)
