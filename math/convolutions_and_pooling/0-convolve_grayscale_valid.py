#!/usr/bin/env python3
"""a functionthat performs a valid convolution on grayscale images """
import numpy as np


def convolve_grayscale_valid(images, kernel):
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
    output_h = (h - kh) + 1
    output_w = (w - kw) + 1
    kernel = np.repeat(kernel[np.newaxis, :, :], m, axis=0)
    output = np.zeros((m, output_h, output_w))
    for i in range(0, output_h):
        for j in range(0, output_w):
            output[:, i, j] = (kernel * images[:, i: i + kh, j: j + kw]).sum(axis=(1, 2))
    return output
