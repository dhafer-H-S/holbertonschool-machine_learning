#!/usr/bin/env python3
"""a functionthat performs a valid convolution on grayscale images """
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    images with shape (m, h, w) containg multiple grayscale images
    m number of images
    h height in pixels of the images
    w is the weidth on pixels of the images

    kernel with shape of (kh, kw) containg the kernel for the convolution
    kh is height of the kernelt
    kw is the width of the kernel
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding
    output_h = h + 2 * ph - kh + 1
    output_w = w + 2 * pw - kw + 1
    output = np.zeros((m, output_h, output_w))
    """calculate the padding needed"""
    pad = ((0, 0), (ph, ph), (pw, pw))
    images = np.pad(images, pad_width=pad, mode='constant')

    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = (kernel * images[:, i: i + kh,
                               j: j + kw]).sum(axis=(1, 2))
    return output
