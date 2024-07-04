#!/usr/bin/env python3
"""4. Frequency"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    plot a histogram of student scores for a project
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    plt.title('Project A')
    plt.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
    plt.xlabel('Grades')
    plt.xlim(0, 100)
    plt.ylabel('Number of Students')
    plt.ylim(0, 30)
    plt.xticks(np.arange(0, 101, 10))
    plt.show()
