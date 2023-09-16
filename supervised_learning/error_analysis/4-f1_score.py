#!/usr/bin/env python3
""" f1 score function """
import numpy as np


sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ f1 score function"""
    sens = sensitivity(confusion)
    prec = precision(confusion)
    f1_scores = 2 * (prec * sens) / (prec + sens)
    return f1_scores
