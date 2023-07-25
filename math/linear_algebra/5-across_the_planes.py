#!/usr/bin/env python3
""" function that add two matrixes """


def add_matrices2D(mat1, mat2):
    """
    Adds two matrices element-wise.
    """
    # Check if the matrices have the same shape
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None

    # Initialize a new matrix to store the sum
    sum_matrix = [[0 for _ in range(len(mat1[0]))] for _ in range(len(mat1))]

    # Iterate through the matrices and add the corresponding elements
    for i in range(len(mat1)):
        for j in range(len(mat1[0])):
            sum_matrix[i][j] = mat1[i][j] + mat2[i][j]

    return sum_matrix
