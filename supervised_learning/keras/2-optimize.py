#!/usr/bin/env python3
"""set up adam optimization for keras model"""
import tensorflow.keras as k


def optimize_model(network, alpha, beta1, beta2):
    """
    network is a model to optimize
    alpha is learning rate
    beta1 is first adam optimization parameter
    beta2 is second adam optimization parameter
    """
    optimizer = k.optimizers.Adam(alpha, beta1, beta2)
    model = network.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
