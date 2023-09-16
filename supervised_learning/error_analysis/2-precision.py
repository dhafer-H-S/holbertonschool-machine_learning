#!/usr/bin/env python3
"""calculates the precision for eache class in confusion matrix """
import numpy as np


def precision(confusion):
    """ find the precision of the neural network """
    classes = confusion.shape[0]
    precision = np.zeros(classes)
    for i in range(classes):
        true_positive = confusion[i, i]
        """
        this represents the number of data points correctly classified
        as this class
        """
        total_actuale_positive = np.sum(confusion[:, i])
        """
        this np.sum(confusion[:, i]) for precision calculation it correctly
        identifies false positive
        """
        """ this represent the total number of data
        this take the entire column of the indexed i from the confusion """
        False_positive = total_actuale_positive - true_positive
        """ this calculates the false positive in  """
        precision_value = true_positive / (true_positive + False_positive)
        precision[i] = precision_value
    return precision
