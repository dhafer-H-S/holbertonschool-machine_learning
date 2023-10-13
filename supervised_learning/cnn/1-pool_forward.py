#!/usr/bin/env python3
"""
function that performs a forward propagation over
 a pooling layer of a neural network
"""
import numpy as np


def pool_forward(A, kernel_shape, stride=(1, 1), mode='max'):
    """
    A is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    kernel_shape is a tuple of (kh, kw) containing the size of the kernel
    for the pooling
        kh is the kernel height
        kw is the kernel width
    stride is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    mode is a string containing either max or avg, indicating whether to
    perform maximum or average pooling, respectively
    """
    m, h_prev, w_prev, c_prv = A.shape
    kh, kw = kernel_shape
    sh, sw = stride
    """
    calculate the output dimensions based oh h_prev, w_prev, kh, kw, sh, sw
    """
    output_h = int((h_prev - kh) / sh + 1)
    output_w = int((w_prev - kw) / sw + 1)
    output = np.zeros((m, output_h, output_w, c_prv))
    for i in range(m):
        for j in range(output_h):
            for k in range(output_w):
                for x in range(c_prv):
                    if mode == 'max':
                        output[i, j, k, x] = np.max(A[i, j * sh: j * sh + kh, k * sw: k * sw + kw, x])
                    elif mode == 'avg':
                        output[i, j, k, x] = np.mean(A[i, j * sh: j * sh + kh, k * sw: k * sw + kw, x])
    return output