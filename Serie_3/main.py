"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2019_12_03

Main program to demonstrate the functionality of our modules.
"""
import block_matrix
import rhs
import linear_solvers
import numpy as np

def f(x):
    return (2*np.pi*(x[1]*np.cos(np.pi*x[0])*np.sin(np.pi*x[1])+ x[0]*np.sin(np.pi*x[0])*(np.cos(np.pi*x[1]) - np.pi*x[1]*np.sin(np.pi*x[1]))))

def u(x):
    return x[0]*x[1]*np.sin(np.pi*x[0])*np.sin(np.pi*x[1])

def main():
    """ Main function to demonstrate the functionality of our modules.
    """
    rhs.plot_error(u, f, 2,range(2, 11))
    block_matrix.plot_non_zeros(range(2, 11))
    block_matrix.plot_cond(range(2, 11))

if __name__ == "__main__":
    main()
