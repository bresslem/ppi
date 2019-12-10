"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2019_12_04

Main program to demonstrate the functionality of our modules.
"""
import block_matrix
import rhs
import numpy as np

def f1(x): #pylint: disable=invalid-name
    """Function f of the Poisson-problem for d = 1
    """
    return -1*(np.pi*(2*np.cos(np.pi*x) - np.pi*x*np.sin(np.pi*x)))

def u1(x): #pylint: disable=invalid-name
    """Function u of the Poisson-problem for d = 1
    """
    return x*np.sin(np.pi*x)

def f2(x): #pylint: disable=invalid-name
    """Function f of the Poisson-problem for d = 2
    """
    return (-2*np.pi*(x[1]*np.cos(np.pi*x[0])*np.sin(np.pi*x[1])+
                      x[0]*np.sin(np.pi*x[0])
                      *(np.cos(np.pi*x[1])- np.pi*x[1]*np.sin(np.pi*x[1]))))

def u2(x): #pylint: disable=invalid-name
    """Function u of the Poisson-problem for d = 2
    """
    return x[0]*np.sin(np.pi*x[0])*x[1]*np.sin(np.pi*x[1])

def f3(x): #pylint: disable=invalid-name
    """Function f of the Poisson-problem for d = 3
    """
    return -1*(np.pi*x[1]*x[2]*(2*np.cos(np.pi*x[0])
                                - np.pi*x[0]*np.sin(np.pi*x[0]))
               *np.sin(np.pi*x[1])*np.sin(np.pi*x[2])+
               np.pi*x[0]*x[2]*(2*np.cos(np.pi*x[1])
                                - np.pi*x[1]*np.sin(np.pi*x[1]))
               *np.sin(np.pi*x[0])*np.sin(np.pi*x[2])+
               np.pi*x[1]*x[0]*(2*np.cos(np.pi*x[2])
                                - np.pi*x[2]*np.sin(np.pi*x[2]))
               *np.sin(np.pi*x[1])*np.sin(np.pi*x[0]))

def u3(x): #pylint: disable=invalid-name
    """Function u of the Poisson-problem for d = 3
    """
    return x[0]*np.sin(np.pi*x[0])*x[1]*np.sin(np.pi*x[1])*x[2]*np.sin(np.pi*x[2])


def main():
    """ Main function to demonstrate the functionality of our modules.
    """
#    rhs.plot_error(u1, f1, 1, range(2, 51))
#    rhs.plot_error(u2, f2, 2, range(2, 11))
#    rhs.plot_error(u3, f3, 3, range(2, 11))
#    block_matrix.plot_non_zeros(range(2, 11))
#    block_matrix.plot_cond(range(2, 11), 3)
    rhs.plot_functions(u2, f2, 51)

if __name__ == "__main__":
    main()
