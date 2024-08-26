#!/usr/bin/env python3

"""function that performs agglomerative clustering"""

import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

def agglomerative(X, dist):
    """
    Performs agglomerative clustering
    with Ward linkage
    and plots a dendrogram.
    Parameters:
    - X: numpy.ndarray of shape (n, d) containing the dataset
    - dist: float, the maximum cophenetic distance for all clusters
    Returns:
    - clss: numpy.ndarray of shape (n,) containing the
    cluster indices for each data point
    """
    # Perform hierarchical/agglomerative clustering using Ward linkage
    linkage_matrix = sch.linkage(X, method='ward')
    # Determine the clusters using the maximum cophenetic distance
    clss = sch.fcluster(linkage_matrix, t=dist, criterion='distance')
    # Plot the dendrogram
    plt.figure()
    sch.dendrogram(linkage_matrix, color_threshold=dist)
    plt.show()

    return clss
