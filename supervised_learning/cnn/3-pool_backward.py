#!/usr/bin/env python3
"""
function that performs back propagation over
a pooling layer of a neural network
"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    function that performs back propagation over
    a pooling layer of a neural network
    """
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)
    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                x = j * sh
                y = k * sw
                """
                x and y are used to compute the starting indices of the slice
                of the inputethat corresponds to the current position of the
                output
                """
                for o in range(c_new):
                    if mode == 'max':
                        slice_A = A_prev[i, x:x + kh, y:y + kw, o]
                        mask = (slice_A == np.max(slice_A))
                        dA_prev[i, x:x + kh, y:y + kw,
                                o] += (dA[i, j, k, o] * mask)
                    elif mode == 'avg':
                        average_dA = dA[i, j, k, o] / (kh * kw)
                        mask = np.ones((kh, kw))
                        dA_prev[i, x:x + kh, y:y + kw, o] += mask * average_dA
    return dA_prev
