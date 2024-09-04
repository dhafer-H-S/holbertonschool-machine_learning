#!/usr/bin/env python3

"""
performs the forward algorithm for a hidden markov model
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Perform the forward algorithm for a Hidden Markov Model.

    Parameters:
    - Observation: numpy.ndarray of shape (T,) containing
    the index of the observation
    - Emission: numpy.ndarray of shape (N, M) containing
    the emission probabilities
    - Transition: numpy.ndarray of shape (N, N) containing
    the transition probabilities
    - Initial: numpy.ndarray of shape (N, 1) containing the
    initial state probabilities

    Returns:
    - P: The likelihood of the observations given the model
    - F: numpy.ndarray of shape (N, T) containing the forward
    path probabilities
    """

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

    T = Observation.shape[0]  # Number of observations
    N, M = Emission.shape      # N hidden states, M possible observations

    if (Emission.shape[1] != np.max(Observation) + 1 or
        Transition.shape[0] != N or
        Transition.shape[1] != N or
        Initial.shape[0] != N or
            Initial.shape[1] != 1):
        return None, None
    # Initialize forward matrix F with zeros
    F = np.zeros((N, T))
    """
    T : number of observations
    N : number of hidden states
    """
    """
    loop through hidden states
    the loop initialize the first column of forward matrix F
    this column corresponds to the probability of starting
    in each hidden state and immediately observing the first
    observation in the sequence
    """
    for H in range(N):
        F[H, 0] = Initial[H, 0] * Emission[H, Observation[0]]
    """loop through observations """
    for o in range(1, T):
        for h in range(N):
            """  """
            F[h, o] = np.sum(F[:, o - 1] * Transition[:, h]) * \
                Emission[h, Observation[o]]
    P = np.sum(F[:, -1])
    return P, F
