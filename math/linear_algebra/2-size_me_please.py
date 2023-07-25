#!/usr/bin/env python3


"""a fonction tha calculate the sape of a matrix and return itt """


def matrix_shape(matrix):
    """matrix_shape = fonction to calculate the sape of matrix"""
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0]
    return shape
