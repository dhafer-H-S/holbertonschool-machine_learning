#!/usr/bin/env python3
"""line"""
import numpy as np
import matplotlib.pyplot as plt


def line():
    """line ploting"""
    y = np.arange(0, 11) ** 3
    print(y)
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(y, color='red')
    plt.axis(0, 10)
    plt.show()
