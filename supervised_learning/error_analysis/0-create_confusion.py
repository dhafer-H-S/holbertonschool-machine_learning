#!/usr/bin/env python3
""" create confusion matrix"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    lables are the correct labels for each data point
    logits are the predicted labels
    """
    classes = labels.shape[1]
    
    # Initialize the confusion matrix with zeros
    confusion = np.zeros((classes, classes), dtype=int)
    
    # Iterate through each data point
    for i in range(labels.shape[0]):
        # Find the index of the correct label (truth) and predicted label
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(logits[i])
        
        # Increment the corresponding cell in the confusion matrix
        confusion[true_label, predicted_label] += 1
    
    return confusion