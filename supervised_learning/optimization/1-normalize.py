#!/usr/bin/env python3
""" standardizes a matrix """


import numpy as np


def normalize(X, m, s):
    normalized_X = (X - m) / s
    return normalized_X
