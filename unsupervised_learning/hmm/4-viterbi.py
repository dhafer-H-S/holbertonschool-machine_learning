#!/usr/bin/env python3
"""
Viterbi Algorithm for Hidden Markov Model
"""

import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    Performs the Viterbi algorithm for a Hidden Markov Model.

    Parameters:
    - Observation: numpy.ndarray of shape (T,) containing
    the index of the observation
    - Emission: numpy.ndarray of shape (N, M) containing
    the emission probabilities
    - Transition: numpy.ndarray of shape (N, N) containing
    the transition probabilities
    - Initial: numpy.ndarray of shape (N, 1) containing
    the initial state probabilities

    Returns:
    - path: List of length T containing the most likely
    sequence of hidden states
    - P: The probability of obtaining the path sequence
    """

    # Check if the inputs are valid
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

    if (Emission.shape[1] != np.max(Observation) + 1 or
            Transition.shape[0] != N or
            Transition.shape[1] != N or
            Initial.shape[0] != N or
            Initial.shape[1] != 1):
        return None, None

    """Initialize variables"""
    V = np.zeros((N, T))  # Viterbi table
    B = np.zeros((N, T), dtype=int)  # Backpointer table

    """Initial probabilities: V[h, 0] = Initial[h] * Emission[h, Observation[0]]"""
    for h in range(N):
        V[h, 0] = Initial[h, 0] * Emission[h, Observation[0]]
        B[h, 0] = 0  # No backpointer for the first observation

    """Recursion: Fill the Viterbi and Backpointer tables"""
    for t in range(1, T):
        for h in range(N):
            # Calculate probability for each previous state
            prob = V[:, t-1] * Transition[:, h]
            # Store the index of the best previous state
            B[h, t] = np.argmax(prob)
            # Choose the max probability path
            V[h, t] = np.max(prob) * Emission[h, Observation[t]]

    """Termination: Find the most probable final state"""
    P = np.max(V[:, T-1])  # Maximum probability in the last column
    last_state = np.argmax(V[:, T-1])  # Index of the most probable last state

    """Backtrack to find the most likely path"""
    path = [last_state]  # Start with the last state
    for t in range(T-1, 0, -1):
        last_state = B[last_state, t]
        """Insert the previous state at the beginning of the path"""
        path.insert(0, last_state)

    return path, P
