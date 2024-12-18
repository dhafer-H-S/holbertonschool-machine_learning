#!/usr/bin/env python3
""" create confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    lables are the correct labels for each data point
    logits are the predicted labels
    """
    """
    labels and logits are of shape (m, classes)
    m stands for data point and
    classes standes for number of classes
    """
    classes = labels.shape[1]
    confusion = np.zeros((classes, classes))
    for i in range(labels.shape[0]):
        correct_labels = np.argmax(labels[i])
        prediction_labels = np.argmax(logits[i])
        """
        the use for np.argmax is to find the maximum value
        in the one hot encoded
        """
        confusion[correct_labels, prediction_labels] += 1
        """
        this ligne take in considireation if the correct labels
        and the prediction are both correct or in an other way it
        keeps track of how many times each true label and predicted label
        combination occurs
        """
    return confusion
