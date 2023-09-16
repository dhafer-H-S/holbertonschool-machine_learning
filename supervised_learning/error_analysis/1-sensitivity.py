#!/usr/bin/env python3
"""
a function that calculates the sensitivity or true positive rate
for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    classes = confusion.shape[0]
    sensitivities = np.zeros(classes)
    """ an array to store sensitivity in it """
    for i in range(classes):
        TP = confusion[i, i]
        """ this make the confusion had access to row and column """
        total_actuale_positive = np.sum(confusion[i, :])
        """ the clone ':' means select all elements along this axis i """
        sensitivity = TP / total_actuale_positive
        """ calculate the sensitivity """
        sensitivities[i] = sensitivity
        """ store sensitivity in the array sensitivities """
    return sensitivities

