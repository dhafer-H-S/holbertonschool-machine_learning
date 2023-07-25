#!/usr/bin/env python3
""" function that add two arrays """


def add_arrays(arr1, arr2):
    """ add two arrays"""
    if len(arr1) != len(arr2):
        return None
    add = []
    for i in range(len(arr1)):
        add.append(arr1[i] + arr2[i])
    return add
