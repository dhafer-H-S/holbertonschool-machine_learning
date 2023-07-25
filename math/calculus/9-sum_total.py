#!/usr/bin/env python3

""" a function thta calculates the summ of a random number"""


def summation_i_squared(n):
    """ summation_i_squared = calculate the squared of a summ """
    if isinstance(n, int) and n >= 1:
        return (n * (n+1) * (2*n+1) // 6)
    else:
        return None
