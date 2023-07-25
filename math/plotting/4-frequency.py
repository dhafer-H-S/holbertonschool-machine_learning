#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)
bins = 10
range = (0, 101)
plt.ylim(0, 30)
plt.xlim(0, 100)
plt.xticks(np.arange(0, 101, step=10))
# plt.hist(student_grades, bins, range, color='blue',histtype='bar' , rwidth=0.8)
plt.hist(student_grades, bins, range, edgecolor='black')
plt.xlabel("Grades")
plt.ylabel("Number of Students")
plt.title("Project A")
plt.show()