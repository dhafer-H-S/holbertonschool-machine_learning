#!/usr/bin/env python3
""" a functiont that calculates weights moving average of a data """


def moving_average(data, beta):
    """ data is the list of data to calcualte the moving average of """
    """ beta is the weight used for the moving average """
    avg = []
    """ avg is a ist used to store the calculated moving average values """
    prev = 0
    """ prev is used to keep track of the previous weighted average """
    for i, d in enumerate(data):
        """
        enumerate is used soo the 'i' trace the index
        and 'd' trace it's value of the index
        """
        prev = (beta * prev + (1 - beta) * d)
        """ calculate the weighted moving average using the prev
        is updated as a weighted sum of the previous value and the
        current data point d """
        correction = prev / (1 - (beta ** (i + 1)))
        """ calculate the correct factor using the formula that
        adjust the moving average to account for the bias introduced by
        the initial values """
        avg.append(correction)
    return avg
