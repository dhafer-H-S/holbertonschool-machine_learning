#!/usr/bin/env python3
"""
function that performs back propagation over
a convolution layer of a neural network
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """Backward propagation for Conv layer."""
    m, h_new, w_new, c_new = dZ.shape
    _, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, _, _ = W.shape
    sh, sw = stride
    output_h = int((h_prev - kh + 2 * (kh // 2)) / sh + 1)
    output_w = int((w_prev - kw + 2 * (kw // 2)) / sw + 1)
    dA_prev = np.zeros_like(A_prev)
    dW = np.zeros_like(W)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    if padding == 'same':
        pad_h = max((output_h - 1) * sh + kh - h_prev, 0)
        pad_w = max((output_w - 1) * sw + kw - w_prev, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        A_prev = np.pad(A_prev, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant')
    else:
        pad_h, pad_w = 0, 0
    for i in range(m):
        for h in range(output_h):
            for w in range(output_w):
                for c in range(c_new):
                    # Compute the slice of A_prev that was used to generate the output
                    slice_A_prev = A_prev[i, h * sh: h * sh + kh, w * sw: w * sw + kw, :]

                    # Update dA_prev using the chain rule
                    dA_prev[i, h * sh: h * sh + kh, w * sw: w * sw + kw, :] += W[:, :, :, c] * dZ[i, h, w, c][..., np.newaxis]

                    # Update dW using the chain rule
                    dW[:, :, :, c] += slice_A_prev * dZ[i, h, w, c][..., np.newaxis]

    # Remove padding from dA_prev if necessary
    if padding == 'same':
        dA_prev = dA_prev[:, pad_top:-pad_bottom, pad_left:-pad_right, :]

    return dA_prev, dW, db
