#!/usr/bin/env python3
"""Minor"""


def minor(matrix):
    """calculates the minor matrix of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix) == 1:
        return [[1]]

    if len(matrix) == 2:
        return [[matrix[1][1], matrix[1][0]], [matrix[0][1], matrix[0][0]]]

    minor = []
    for idx, element in enumerate(matrix[0]):
        inner = []
        for row in matrix[1:]:
            inner.append(row[:idx] + row[idx + 1:])
        minor.append(inner)

    return minor