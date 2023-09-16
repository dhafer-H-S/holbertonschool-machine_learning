#!/usr/bin/env python3
""" f1 score function """
import numpy as np


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    classes = confusion.shape[0]
    f1_scores = np.zeros(classes)
    for i in range(classes):
        sens = sensitivity(confusion[i, :])
        prec = precision(confusion[:, i])
        f1_scores[i] = 2 * (prec * sens) / (prec + sens)
    return f1_scores
