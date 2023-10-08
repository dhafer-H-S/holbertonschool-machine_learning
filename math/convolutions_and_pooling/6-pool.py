#!/usr/bin/env python3
"""
a function performs pooling on images
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    images of shape (m, h, w, c)
    kernel of shape (kh, kw)
    stride of shape (sh, sw)
    mode indicates the type of pooling
        max indicates max pooling
        avg indicates average pooling
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    output_h = int((h - kh) / sh + 1)
    output_w = int((w - kw) / sw + 1)
    output = np.zeros((m, output_h, output_w, c))
    for i in range(0, output_h):
        x = i * sh
        for j in range(0, output_w):
            y = j * sw
            if mode == 'max':
                output[:, i, j, :] = np.max(
                    images[:, x:x + kh, y:y + kw, :], axis=(1, 2))
            elif mode == 'avg':
                output[:, i, j, :] = np.average(
                    images[:, x:x + kh, y:y + kw, :], axis=(1, 2))
    return output
