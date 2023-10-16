#!/usr/bin/env python3
"""
function that performs back propagation over
a convolution layer of a neural network
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new)
    containing the partial derivatives with respect to the unactivated
    output of the convolutional layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output
    A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
    containing the output of the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing
    the kernels for the convolution
        kh is the filter height
        kw is the filter width
        c_prev is the number of channels in the previous layer
        c_new is the number of channels in the output
    b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
    applied to the convolution
    padding is a string that is either same or valid, indicating the type
    of padding used
    stride is a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width
    """
    m, prev_h, prev_w, _ = A_prev.shape
    m, new_h, new_w, new_c = dZ.shape
    sh, sw = stride
    kh, kw, _, new_c = W.shape
    if padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = int(np.ceil((sh*(prev_h-1)-prev_h+kh)/2))
        pw = int(np.ceil((sw*(prev_w-1)-prev_w+kw)/2))
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    A_prev = np.pad(A_prev, pad_width=npad, mode='constant')
    dw = np.zeros_like(W)
    dA = np.zeros_like(A_prev)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    for img in range(m):
        for h in range(new_h):
            for w in range(new_w):
                x = h * sh
                y = w * sw
                for f in range(new_c):
                    filt = W[:, :, :, f]
                    dz = dZ[img, h, w, f]
                    slice_A = A_prev[img, x:x+kh, y:y+kw, :]
                    dw[:, :, :, f] += slice_A * dz
                    dA[img, x:x+kh, y:y+kw, :] += dz * filt

    if padding == 'same':
        dA = dA[:, ph:-ph, pw:-pw, :]
    return dA, dw, 
