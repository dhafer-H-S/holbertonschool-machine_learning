#!/usr/bin/env python3
""" a convolution function performs on image with diffrent or multiple types of kernels """
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    
    images of shape (m, h, w, c)
    m is the number of images
    h is the height in pixels of the images
    w is the width in pixels of the images
    c is the number of channels in the image
    kernel of shape (kh, kw, nc)
    kh is the height of the kernel
    kw is the width of the kernel
    nc is the number of kernels
    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
    if ‘same’, performs a same convolution
    if ‘valid’, performs a valid convolution
    if a tuple:
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    the image should be padded with 0’s
    stride is a tuple of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    """
    m, h, w, c = images.shape
    kh, kw, c, nc = kernels.shape
    sh, sw = stride

    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = int(np.ceil((sh * (h - 1) - h + kh) / 2))
        pw = int(np.ceil((sw * (w - 1) - w + kw) / 2))
    output_h = int((h + 2 * ph - kh) / sh + 1)
    output_w = int((w + 2 * pw - kw) / sw + 1)
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    images = np.pad(images, pad_width=npad, mode='constant')
    output = np.zeros((m, output_h, output_w, nc))
    for i in range(0, output_h):
        x = i * sh
        for j in range(0, output_w):
            y = j * sw
            for k in range(nc):
                output[:, i, j, k] = np.sum(
                    images[:, x:x+kh, y:y+kw] * kernels[:, :, :, k],
                    axis=(1, 2, 3))
    return output
