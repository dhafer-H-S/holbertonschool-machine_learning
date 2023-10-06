#!/usr/bin/env python3
""" performs convolution om inmages with channels """
import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with channels.

    Args:
        images (numpy.ndarray): Input images with shape (m, h, w, c).
        kernel (numpy.ndarray): Convolution kernel with shape (kh, kw, c).
        padding (tuple or str): Padding for the height and width dimensions. 
            Can be a tuple of (ph, pw), 'same', or 'valid'.
        stride (tuple): Stride for the height and width dimensions.

    Returns:
        numpy.ndarray: Convolved images with shape (m, output_h, output_w, c).
    """
    kh, kw, _ = kernel.shape
    sh, sw = stride
    m, h, w, _ = images.shape
    if type(padding) is tuple:
        ph, pw = padding
    elif padding == 'valid':
        ph, pw = 0, 0
    else:
        ph = int(np.ceil((sh*(h-1)-h+kh)/2))
        pw = int(np.ceil((sw*(w-1)-w+kw)/2))
    oh = int((h+2*ph-kh)/sh+1)
    ow = int((w+2*pw-kw)/sw+1)
    npad = ((0, 0), (ph, ph), (pw, pw), (0, 0))
    images = np.pad(images, pad_width=npad, mode='constant')
    output = np.zeros((m, oh, ow))
    for i in range(0, oh):
        x = i * sh
        for j in range(0, ow):
            y = j * sw
            output[:, i, j] = np.sum(images[:, x:x+kh, y:y+kw] * kernel,
                                     axis=(1, 2, 3))

    return output
