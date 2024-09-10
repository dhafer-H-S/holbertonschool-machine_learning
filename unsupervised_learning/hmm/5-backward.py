#!/usr/bin/env python3
"""
Backward Algorithm for Hidden Markov Model
"""

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    Performs the Backward algorithm for a Hidden Markov Model.

    Parameters:
    - Observation: numpy.ndarray of shape (T,) containing the index
    of the observation
    - Emission: numpy.ndarray of shape (N, M) containing
    the emission probabilities
    - Transition: numpy.ndarray of shape (N, N) containing
    the transition probabilities
    - Initial: numpy.ndarray of shape (N, 1) containing
    the initial state probabilities

    Returns:
    - P: The likelihood of the observations given the model
    - B: numpy.ndarray of shape (N, T) containing the backward path
    probabilities
    """

    # Validate inputs
    if (not isinstance(Observation, np.ndarray) or
        not isinstance(Emission, np.ndarray) or
        not isinstance(Transition, np.ndarray) or
            not isinstance(Initial, np.ndarray)):
        return None, None

    if (len(Observation.shape) != 1 or
        len(Emission.shape) != 2 or
        len(Transition.shape) != 2 or
            len(Initial.shape) != 2):
        return None, None

    # Number of observations
    T = Observation.shape[0]
    # N hidden states, M possible observations
    N, M = Emission.shape

    # Check if dimensions match the expectations
    if (Emission.shape[1] != np.max(Observation) + 1 or
        Transition.shape[0] != N or
        Transition.shape[1] != N or
        Initial.shape[0] != N or
            Initial.shape[1] != 1):
        return None, None

    # Initialize the backward probabilities matrix B with shape (N, T)
    B = np.zeros((N, T))

    # Step 1: Initialize B at the last observation to 1 (Base case for
    # recursion)
    B[:, T - 1] = 1

    # Step 2: Recursively calculate backward probabilities
    for t in range(
            # Iterate backwards from the second last observation
            T - 2, -1, -1):
        for i in range(N):  # Iterate over each hidden state
            B[i, t] = np.sum(B[:, t + 1] * Transition[i, :]
                             * Emission[:, Observation[t + 1]])

    # Step 3: Calculate the total probability of the observation sequence
    P = np.sum(Initial[:, 0] * Emission[:, Observation[0]] * B[:, 0])

    return P, B
