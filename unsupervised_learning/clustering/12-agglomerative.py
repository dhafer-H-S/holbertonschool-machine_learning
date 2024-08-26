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
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    sch.dendrogram(linkage_matrix, color_threshold=dist)
    plt.title('Agglomerative Clustering Dendrogram')
    plt.xlabel('Sample index')
    plt.ylabel('Distance')
    plt.show()
    # Determine the clusters using the maximum cophenetic distance
    clss = sch.fcluster(linkage_matrix, t=dist, criterion='distance')
    return clss
