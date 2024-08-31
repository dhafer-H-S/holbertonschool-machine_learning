#!/usr/bin/env python3

"""
function that determines the probability of a matrix of a markov
chain being in a particular state after a specified number of iterations
"""
import numpy as np


def markov_chain(P, s, t=1):
    """
    p: represent the transition matrix of shape (n, n)
    P[i, j] is the probability of transition from state i to j
    n : number of states in the markov chain
    s: shape (1, n) representing the probability of starting in each state
    t : number of iteration that the markov chain has been through
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return None
    if not isinstance(s, np.ndarray) or len(s.shape) != 2:
        return None
    if P.shape[0] != P.shape[1] or P.shape[0] != s.shape[1]:
        return None
    if t <= 0:
        return None
    state_distribution = s
    for _ in range(t):
        state_distribution = np.dot(state_distribution, P)
    return state_distribution