#!/usr/bin/env python3
""" a functiont that calculates weights moving average of a data """


def moving_average(data, beta):
    """ data is the list of data to calcualte the moving average of"""
    """ beta is the weight used for the moving average """
    avg = []
    prev = 0
    for i, d in enumerate(data):
        prev = (beta * prev + (1 - beta) * d)
        correction = prev / (1 - (beta ** (i + 1)))
        avg.append(correction)
    return avg
