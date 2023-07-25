#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)
plt.ylim(0, 1)
plt.xlim(0, 20000)

lines = plt.plot(x, y1, 'r--', label='C-14')
lines = plt.plot(x, y2, 'g-', label='Ra-226')

plt.xlabel("Time (years)")
plt.ylabel("Fraction Remaining")
plt.title("Exponential Decay of Radioactive Elements")
plt.plot(x, y1, 'r--', x, y2, 'g-')
plt.legend(bbox_to_anchor=(0.77, 1), loc=2, borderaxespad=0.5)
plt.show()