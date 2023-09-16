#!/usr/bin/env python3
""" calculates the specificity for each class in a confusion matrix """
import numpy as np


def specificity(confusion):
    """
    calcule specificity specifity = TN / TN + FP
    """
    classes = confusion.shape[0]
    specificties = np.zeros(classes)
    for i in range(classes):
        True_Positive = confusion[i, i]
        total_predicted_values = np.sum(confusion[:, i])
        total_correct_value = np.sum(confusion[i, :])
        false_positive = total_predicted_values - True_Positive
        false_negative = total_correct_value - True_Positive
        True_negative = np.sum(confusion) - \
            (True_Positive + false_positive + false_negative)
        specificity = True_negative / (True_negative + false_positive)
        specificties[i] = specificity
    return specificties
