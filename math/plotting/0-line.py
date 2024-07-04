#!/usr/bin/env python3
"""0. Line Graph"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """plot y as a line graph"""
    y = np.arange(0, 11) ** 3
    print(y)
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y, color='red')
    plt.xlim(0, 10)
    plt.show()
