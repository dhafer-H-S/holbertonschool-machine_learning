#!/usr/bin/env python3
"""a functionthat performs a valid convolution on grayscale images """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    images with shape (m, h, w) containg multiple grayscale images
    m number of images
    h height in pixels of the images
    w is the weidth on pixels of the images

    kernel with shape of (kh, kw) containg the kernel for the convolution
    kh is height of the kernelt
    kw is the width of the kernel


    padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
    if ‘same’, performs a same convolution
    if ‘valid’, performs a valid convolution
    if a tuple:
    ph is the padding for the height of the image
    pw is the padding for the width of the image
    the image should be padded with 0’s


    stride with shape of (sh, sw)
    sh is the stride for the height of the image
    sw is the stride for the width of the image
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(np.ceil(((h - 1) * sh + kh - h) / 2))
        pw = int(np.ceil(((w - 1) * sw + kw - w) / 2))
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph, pw = padding

    """calculate the padding needed"""
    pad = ((0, 0), (ph, ph), (pw, pw))
    images = np.pad(images, pad_width=pad, mode='constant')

    output_h = int(((h + 2 * ph - kh) / sh) + 1)
    output_w = int(((w + 2 * pw - kw) / sw) + 1)
    output = np.zeros((m, output_h, output_w))

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (kernel * images[:, i*sh: i*sh + kh,
                               j*sw: j*sw + kw]).sum(axis=(1, 2))
    return output
