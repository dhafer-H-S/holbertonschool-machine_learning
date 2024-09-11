#!/usr/bin/env python3
"""
Perform the forward algorithm for a Hidden Markov Model.
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Perform the forward algorithm for a Hidden Markov Model.

    Parameters:
    - Observation: numpy.ndarray of shape (T,) containing the index of the observation
    - Emission: numpy.ndarray of shape (N, M) containing the emission probabilities
    - Transition: numpy.ndarray of shape (N, N) containing the transition probabilities
    - Initial: numpy.ndarray of shape (N, 1) containing the initial state probabilities

    Returns:
    - alpha: numpy.ndarray of shape (N, T) containing the forward probabilities
    - P: The probability of the observations given the model
    """
    T = Observation.shape[0]
    N = Emission.shape[0]

    alpha = np.zeros((N, T))
    alpha[:, 0] = Initial.flatten() * Emission[:, Observation[0]]

    for t in range(1, T):
        for i in range(N):
            alpha[i, t] = np.sum(alpha[:, t-1] * Transition[:, i]) * Emission[i, Observation[t]]

    P = np.sum(alpha[:, -1])
    return alpha, P

def backward(Observation, Emission, Transition, Initial):
    """
    Perform the backward algorithm for a Hidden Markov Model.

    Parameters:
    - Observation: numpy.ndarray of shape (T,) containing the index of the observation
    - Emission: numpy.ndarray of shape (N, M) containing the emission probabilities
    - Transition: numpy.ndarray of shape (N, N) containing the transition probabilities
    - Initial: numpy.ndarray of shape (N, 1) containing the initial state probabilities

    Returns:
    - beta: numpy.ndarray of shape (N, T) containing the backward probabilities
    - P: The probability of the observations given the model
    """
    T = Observation.shape[0]
    N = Emission.shape[0]

    beta = np.zeros((N, T))
    beta[:, -1] = 1

    for t in range(T-2, -1, -1):
        for i in range(N):
            beta[i, t] = np.sum(Transition[i, :] * Emission[:, Observation[t+1]] * beta[:, t+1])

    alpha, P = forward(Observation, Emission, Transition, Initial)
    return beta, P

def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a Hidden Markov Model.

    Parameters:
    - Observations: numpy.ndarray of shape (T,) containing the index of the observation
    - Transition: numpy.ndarray of shape (M, M) containing the initialized transition probabilities
    - Emission: numpy.ndarray of shape (M, N) containing the initialized emission probabilities
    - Initial: numpy.ndarray of shape (M, 1) containing the initialized starting probabilities
    - iterations: Number of times expectation-maximization should be performed

    Returns:
    - Transition: The converged transition probabilities
    - Emission: The converged emission probabilities
    """
    T = Observations.shape[0]
    M, N = Emission.shape

    for _ in range(iterations):
        # E-step: Calculate alpha and beta
        alpha, _ = forward(Observations, Emission, Transition, Initial)
        beta, _ = backward(Observations, Emission, Transition, Initial)
        
        # Calculate gamma (probability of being in state i at time t)
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=0, keepdims=True)
        
        # Calculate xi (probability of being in state i at time t and state j at time t+1)
        xi = np.zeros((M, M, T-1))
        for t in range(T-1):
            xi[:, :, t] = (alpha[:, t][:, np.newaxis] * Transition * 
                           Emission[:, Observations[t+1]] * beta[:, t+1])
            xi[:, :, t] /= np.sum(xi[:, :, t])

        # M-step: Update the transition and emission probabilities
        Transition = np.sum(xi, axis=2)
        Transition /= np.sum(Transition, axis=1, keepdims=True)
        
        for i in range(M):
            for j in range(N):
                Emission[i, j] = np.sum(gamma[i, Observations == j]) / np.sum(gamma[i, :])
        
        Initial = gamma[:, 0].reshape(-1, 1)
    
    return Transition, Emission
