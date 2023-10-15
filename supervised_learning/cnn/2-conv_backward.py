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
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    output_h = int((h_prev - kh) / sh + 1)
    output_w = int((w_prev - kw) / sw + 1)
    output = np.zeros((m, output_h, output_w, c_prev))
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.zeros_like(b)
    if padding == 'same':
        pad_h = int(np.ceil(((h_prev - 1) * sh + kh - h_prev) / 2))
        pad_w = int(np.ceil(((w_prev - 1) * sw + kw - w_prev) / 2))
        dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]
    else:
        pad_h = 0
        pad_w = 0
    for i in range(m):
        for h in range(output_h):
            for w in range(output_w):
                for c in range(c_new):
                    # Compute the slice of A_prev that was used to generate the
                    # output
                    slice_A_prev = A_prev[i, h * sh: h *
                                          sh + kh, w * sw: w * sw + kw, :]

                    # Update dA_prev using the chain rule
                    dA_prev[i, h * sh: h * sh + kh, w * sw: w * sw + kw,
                        :] += W[:, :, :, c] * dZ[i, h, w, c][..., np.newaxis]

                    # Update dW using the chain rule
                    dW[:, :, :, c] += slice_A_prev * \
                        dZ[i, h, w, c][..., np.newaxis]

                    # Update db using the chain rule
                    db[:, :, :, c] += dZ[i, h, w, c][..., np.newaxis]

        # Remove padding from dA_prev if necessary
        if padding == 'same':
            dA_prev = dA_prev[:, pad_h:-pad_h, pad_w:-pad_w, :]

    return dA_prev, dW, db
