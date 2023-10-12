#!/usr/bin/env python3
"""
a function that performs forword propagation
over a convolution layer of aneural network
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        m is the number of examples
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing kernels
    for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    activation is an activation function applied to the convolution
    padding is a string that is either same or valid, indicating the type of padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    """
    def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
        m, h_prev, w_prev, c_prev = A_prev.shape
        kh, kw, c_prev, c_new = W.shape
        sh, sw = stride
        h_new = int(((h_prev - kh + (2 * padding)) / sh) + 1)
        w_new = int(((w_prev - kw + (2 * padding)) / sw) + 1)
        if padding =='valid':
            ph, pw = 0, 0
        else:
            ph = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
            pw = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
            """
            np.ceil is used to for example we have 4.1 we only get 4 as out put
            """
        npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
        A_prev_padded = np.pad(A_prev, pad_width=npad, mode='constant', constant_values=0)
        output = np.zeros((m, h_new, w_new, c_new))
        for i in range(m):
            for j in range(h_new):
                for k in range(w_new):
                    for l in range(c_new):
                        output[i, j, k, l] = np.sum(
                            A_prev_padded[i, j * sh: j * sh + kh,
                                k * sw: k * sw + kw, :] *
                            W[:, :, :, l])
        A = np.nan_to_num(output, nan=0) + b
        return activation(A)
