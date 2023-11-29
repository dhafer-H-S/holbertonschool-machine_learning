#!/usr/bin/env python3
"""Minor"""


def determinant(matrix):
    """calculates the determinant of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")

    if len(matrix[0]) == 0:
        return 1

    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    det = 0
    for idx, element in enumerate(matrix[0]):
        inner = []
        for row in matrix[1:]:
            inner.append(row[:idx] + row[idx + 1:])
        sign = (-1) ** idx
        det += sign * element * determinant(inner)

    return det


def minor(matrix):
    """calculates the minor matrix of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]

    minor_output = []

    for i in range(len(matrix)):
        inside = []
        for j in range(len(matrix[0])):
            matrix_copy = [row[:] for row in matrix]
            del matrix_copy[i]
            for row in matrix_copy:
                del row[j]
            inside.append(determinant(matrix_copy))
        minor_output.append(inside)

    return minor_output


def cofactor(matrix):
    """calculate the cofator matrix of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) == 1:
        return [[1]]
    cofactor_output = []
    for i in range(len(matrix)):
        inside = []
        for j in range(len(matrix[0])):
            matrix_copy = [row[:] for row in matrix]
            del matrix_copy[i]
            for row in matrix_copy:
                del row[j]
            inside.append(((-1) ** (i + j)) * determinant(matrix_copy))
        cofactor_output.append(inside)
    return cofactor_output


def adjugate(matrix):
    """calculate the adjugate matrix of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    for sub_list in matrix:
        if not isinstance(sub_list, list):
            raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    cofactor_output = cofactor(matrix)
    adjugate_output = []
    for i in range(len(cofactor_output)):
        inside = []
        for j in range(len(cofactor_output[0])):
            inside.append(cofactor_output[j][i])
        adjugate_output.append(inside)
    return adjugate_output


def inverse(matrix):
    """calculating the inverse of a matrix"""
    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")
    if len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    if not isinstance(matrix[0], list):
        raise TypeError("matrix must be a list of lists")
    if len(matrix) != len(matrix[0]):
        raise ValueError("matrix must be a non-empty square matrix")
    else:
        adjugate_output = adjugate(matrix)
        inverse_output = []
        det = determinant(matrix)
        if det == 0:
            return None
        else:
            for i in range(len(adjugate_output)):
                inside = []
                for j in range(len(adjugate_output[0])):
                     inside.append(adjugate_output[i][j] / det)
                inverse_output.append(inside)
            return inverse_output
    