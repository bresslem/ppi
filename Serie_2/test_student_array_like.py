"""
Testfaelle für die Funktion rhs aus Serie 2b
author: Henrik Schneider
2019
"""

import numpy as np
from rhs import rhs

TOL = 10e-14


def f(x):
    return np.sin(x)


def g(x):
    return np.exp(x[0]) + x[1]


def h(x):
    return x[0]**2 + x[1] + x[2]**3


data_1D = np.load('1D.npy')
data_2D = np.load('2D.npy')
data_3D = np.load('3D.npy')

print("1D Test:  ", TOL > np.linalg.norm(rhs(1, 10, f) - data_1D, np.inf))

print("2D Test:  ", TOL > np.linalg.norm(rhs(2, 20, g) - data_2D, np.inf))

print("3D Test:  ", TOL > np.linalg.norm(rhs(3, 13, h) - data_3D, np.inf))
