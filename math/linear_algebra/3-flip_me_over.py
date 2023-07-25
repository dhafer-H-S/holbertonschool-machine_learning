#!/usr/bin/env python3
""" a function that transpose a matrix """


def matrix_transpose(matrix):
    """ transpose a matrix """
    rows = len(matrix)
    cols = len(matrix[0])

    transpose = [[matrix[j][i] for j in range(rows)] for i in range(cols)]
    return transpose
