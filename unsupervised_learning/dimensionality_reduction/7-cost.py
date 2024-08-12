#!/usr/bin/env python3

"""
a function that calculate looss or cost in T-sne
"""

import numpy as np

def cost(P, Q):
    """Ensure no division by zero or log of zero"""
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)

    """Calculate the cost function (KL divergence)"""
    C = np.sum(P * np.log(P / Q))

    return C
