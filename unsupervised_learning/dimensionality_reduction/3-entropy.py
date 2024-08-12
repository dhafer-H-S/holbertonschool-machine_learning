#!/usr/bin/env python3


import numpy as np

"""
calculate the entropy in t-sne
"""

def HP(Di, beta):
    """
     you're essentially quantifying
    how spread out or concentreated the similarities are between
    one point and all other points this helps to control how
    the algorithms represnts the local structure of the data in
    the lower dimensional space
    """
    """
    compute the affinities 
    """
    Pi = np.exp(-Di * beta)
    sum_Pi = np.sum(Pi)
    Pi /= sum_Pi
    """
    compute shannon entropy
    """
    Hi = -np.sum(Pi * np.log2(Pi))
    return Hi, Pi