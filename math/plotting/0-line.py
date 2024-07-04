#!/usr/bin/env python3
"""Line Graph"""

import numpy as np
import matplotlib.pyplot as plt

def line():
    """plot y as a line graph"""
    x = np.arange(0, 11)
    plt.figure(figsize=(6.4, 4.8))
    y = x ** 3
    plt.plot(x, y, color='red')
    plt.xlim(0, 10)
    plt.ylim(0, 1000)
    plt.xlabel('x-axis')
    plt.ylabel('y-axis')
    plt.title('Cubic Line Graph')
    plt.grid(True)
    plt.show()
