#!/usr/bin/env python3
"""_summary_

    Returns:
        _type_: _description_
"""

import numpy as np


def kmeans(X, k, iterations=1000):
    """_summary_

    Args:
        X (_type_): _description_
        k (_type_): _description_
        iterations (int, optional): _description_. Defaults to 1000.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None

    n, d = X.shape

    if not isinstance(k, int) or k <= 0:
        return None, None

    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    C = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), (k, d))

    for _ in range(iterations):
        C_copy = np.copy(C)

        D = np.linalg.norm(X - C[:, np.newaxis], axis=2)

        clss = np.argmin(D, axis=0)

        for j in range(k):
            if len(X[clss == j]) == 0:
                C[j] = np.random.uniform(np.min(X, axis=0),
                                         np.max(X, axis=0), (1, d))
            else:
                C[j] = np.mean(X[clss == j], axis=0)

        if np.array_equal(C, C_copy):
            return C, clss

    D = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    clss = np.argmin(D, axis=0)
    return C, clss
