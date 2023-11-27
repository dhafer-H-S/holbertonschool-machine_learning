#!/usr/bin/env python3
"""Determinant"""


def determinant(matrix):
    """Calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if not all([isinstance(row, list) for row in matrix]):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]) and len(matrix[0]) > 1:
        raise ValueError("matrix must be a square matrix")
    if len(matrix[0]) == 1:
        return matrix[0][0]
    if len(matrix[0]) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for i, k in enumerate(matrix[0]):
        rows = [row for row in matrix[1:]]
        aux_matrix = [[row[n] for n in range(
            len(matrix)) if n != i] for row in rows]
        det += k * (-1) ** i * determinant(aux_matrix)
    return det
