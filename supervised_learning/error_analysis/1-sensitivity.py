#!/usr/bin/env python3
"""
a function that calculates the sensitivity or true positive rate
for each class in a confusion matrix
"""
import numpy as np


def sensitivity(confusion):
    """ calculates sensitivity based on the data in the confusion martix """
    classes = confusion.shape[0]
    """
    confusion is a shape of (classes , classes) and by
    confussion.shape[0] we only get the classes at first not the second
    """
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
