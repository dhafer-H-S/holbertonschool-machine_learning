#!/usr/bin/env python3

"""
This module creates a scatter plot to visualize mountain elevation.
It uses matplotlib to plot x and y coordinates with color coding to
represent the elevation.
"""

import numpy as np
import matplotlib.pyplot as plt


def gradient():
    """
    Plots a gradient scatter plot of mountain elevation.

    This function generates random x and y coordinates, and
    calculates the corresponding
    elevation values based on the distance from the origin.
    It then plots a scatter plot
    of the coordinates, with the color of each point representing
    the elevation.
    Args:
        None
    Returns:
        None
    """
    np.random.seed(5)
    x = np.random.randn(2000) * 10
    y = np.random.randn(2000) * 10
    z = np.random.rand(2000) + 40 - np.sqrt(np.square(x) + np.square(y))

    plt.figure(figsize=(6.4, 4.8))
    plt.title("Mountain Elevation")
    plt.xlabel("x coordinate (m)")
    plt.ylabel("y coordinate (m)")
    scatter = plt.scatter(x, y, c=z, cmap='viridis')
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('elevation (m)')  # Labeling the colorbar

    plt.show()
