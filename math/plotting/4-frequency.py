#!/usr/bin/env python3
"""import modules"""
import numpy as np
import matplotlib.pyplot as plt


def frequency():
    """
    Plot a histogram of student grades.

    This function generates a random set of student grades and plots a histogram
    to visualize the distribution of grades.

    Args:
        None

    Returns:
        None
    """
    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)
    plt.figure(figsize=(6.4, 4.8))
    bins_range = np.arange(start=int(min(student_grades) / 10) * 10,
                           stop=(int(max(student_grades) / 10) + 1) * 10,
                           step=10)
    counts, _ = np.histogram(student_grades, bins=bins_range)
    plt.bar(bins_range[:-1], counts, width=9, edgecolor='black', align='edge')
    plt.xlabel("Grades")
    plt.ylabel("Number of Students")
    plt.title("Project A")
    plt.xticks(bins_range)
    plt.show()

frequency()