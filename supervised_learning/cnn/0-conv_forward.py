#!/usr/bin/env python3
"""
a function that performs forword propagation
over a convolution layer of aneural network
"""
import numpy as np


def conv_forward(A, W, b, activation, padding="same", stride=(1, 1)):
    """
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
        containing the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
        kernels for the convolution
            kh is the filter height
            kw is the filter width
            c_prev is the number of channels in the previous layer
            c_new is the number of channels in the output
        b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
        applied to the convolution
        activation is an activation function applied to the convolution
        padding is a string that is either same or valid, indicating the type
        of padding used
        stride is a tuple of (sh, sw) containing the strides for the
        convolution
            sh is the stride for the height
            sw is the stride for the width
    """
    m, h_prev, w_prev, c_prev = A.shape
    kh, kw, _, c_new = W.shape
    sh, sw = stride

    if padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = int(np.ceil((h_prev * sh - sh + kh - h_prev) / 2))
        pw = int(np.ceil((w_prev * sw - sw + kw - w_prev) / 2))
    h_new = int((h_prev + 2 * ph - kh) / sh) + 1
    w_new = int((w_prev + 2 * pw - kw) / sw) + 1
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    A_prev_padded = np.pad(
        A,
        pad_width=npad,
        mode='constant',
        constant_values=0)
    output = np.zeros((m, h_new, w_new, c_new))

    for i in range(m):
        for j in range(h_new):
            for k in range(w_new):
                for l in range(c_new):
                    output[i, j, k, l] = np.sum(
                        A_prev_padded[i, j * sh: j * sh + kh,
                                      k * sw: k * sw + kw, :] * W[:, :, :, l])
    A = activation(output + b)

    return A
