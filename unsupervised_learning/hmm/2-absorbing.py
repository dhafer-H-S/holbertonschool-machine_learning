#!/usr/bin/env python3

"""
function that absorbing that determine if a markov chain is absorbing
to check if every step is absorbed or not
"""

import numpy as np


def absorbing(P):
    """
    p(n, n)
    p[i, j]
    n : number of state in the new markov chain
    """
    if not isinstance(P, np.ndarray) or P.shape[0] != P.shape[1]:
        return False

    n = P.shape[0]

    # Identify absorbing states
    absorbing_states = np.where(np.diag(P) == 1)[0]

    if len(absorbing_states) == 0:
        return False

    # Partition the matrix
    non_absorbing_states = np.setdiff1d(np.arange(n), absorbing_states)

    if len(non_absorbing_states) == 0:
        return True

    Q = P[np.ix_(non_absorbing_states, non_absorbing_states)]
    identity_matrix = np.eye(Q.shape[0])

    try:
        # Check if I - Q is invertible
        np.linalg.inv(identity_matrix - Q)
    except np.linalg.LinAlgError:
        return False

    return True
