#!/usr/bin/env python3
"""
Baum-Welch Algorithm for Hidden Markov Model
"""

import numpy as np
from scipy.special import logsumexp


def forward(Observations, Emission, Transition, Initial):
    """
    Performs the forward algorithm to calculate alpha values for HMM.
    """
    T = Observations.shape[0]
    M = Transition.shape[0]

    alpha = np.zeros((M, T))
    alpha[:, 0] = Initial[:, 0] * Emission[:, Observations[0]]

    for t in range(1, T):
        for j in range(M):
            alpha[j,
                  t] = np.sum(alpha[:,
                                    t - 1] * Transition[:,
                                                        j]) * Emission[j,
                                                                       Observations[t]]

    return alpha


def backward(Observations, Emission, Transition, Initial):
    """
    Performs the backward algorithm to calculate beta values for HMM.
    """
    T = Observations.shape[0]
    M = Transition.shape[0]

    beta = np.zeros((M, T))
    beta[:, T - 1] = 1

    for t in range(T - 2, -1, -1):
        for i in range(M):
            beta[i, t] = np.sum(
                Transition[i, :] * Emission[:, Observations[t + 1]] * beta[:, t + 1])

    return beta


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a Hidden Markov Model.

    Parameters:
    - Observations: numpy.ndarray of shape (T,) that contains the index of the observation
    - Transition: numpy.ndarray of shape (M, M) containing the initialized transition probabilities
    - Emission: numpy.ndarray of shape (M, N) containing the initialized emission probabilities
    - Initial: numpy.ndarray of shape (M, 1) containing the initialized starting probabilities
    - iterations: number of iterations for expectation-maximization

    Returns:
    - Updated Transition and Emission matrices, or None, None on failure
    """
    T = Observations.shape[0]
    M, N = Emission.shape

    for _ in range(iterations):
        # E-Step: Compute forward and backward probabilities
        alpha = forward(Observations, Emission, Transition, Initial)
        beta = backward(Observations, Emission, Transition, Initial)

        # Calculate xi and gamma
        xi = np.zeros((M, M, T - 1))
        gamma = np.zeros((M, T))

        for t in range(T - 1):
            denom = np.dot(np.dot(
                alpha[:, t].T, Transition) * Emission[:, Observations[t + 1]].T, beta[:, t + 1])
            for i in range(M):
                numer = alpha[i, t] * Transition[i, :] * \
                    Emission[:, Observations[t + 1]].T * beta[:, t + 1].T
                xi[i, :, t] = numer / denom

        # Gamma is the sum over xi for each state
        gamma = np.sum(xi, axis=1)

        # Also need to account for gamma_T-1
        gamma[:, T - 1] = alpha[:, T - 1] / np.sum(alpha[:, T - 1])

        # M-Step: Update the transition, emission, and initial probabilities
        Transition = np.sum(xi, axis=2) / \
            np.sum(gamma[:, :-1], axis=1, keepdims=True)

        for obs in range(N):
            mask = Observations == obs
            Emission[:, obs] = np.sum(
                gamma[:, mask], axis=1) / np.sum(gamma, axis=1)

        Initial = gamma[:, 0].reshape(-1, 1)

    return Transition, Emission
