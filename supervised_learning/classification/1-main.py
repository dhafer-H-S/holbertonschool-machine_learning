#!/usr/bin/env python3

import numpy as np
Neuron = __import__('1-neuron').Neuron

np.random.seed(2)
nx = np.random.randint(100, 1000)
nn = Neuron(nx)
try:
    nn.W = 1
    print('FAIL')
except AttributeError:
    pass
try:
    nn.b = 1
    print('FAIL')
except AttributeError:
    pass
try:
    nn.A = 1
    print('FAIL')
except AttributeError:
    pass
print('OK', end='')