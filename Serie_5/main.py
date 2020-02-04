"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2020_01_03

Main program to demonstrate the functionality of our modules.
"""

#pylint: disable=no-member, unused-import

import block_matrix
import rhs
import linear_solvers
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12

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


def get_cond_matrix(a): #pylint: disable=invalid-name
    """ Computes the condition number of the matrix a.

    Returns
    -------
    float
        condition number with respect to the row sum norm
    """
    return np.linalg.norm(a, np.inf)*np.linalg.norm(np.linalg.inv(a), np.inf)


def print_cond_hilbert(n_list, d): #pylint: disable=invalid-name
    """
    Calculates the condition of the Hilbert-matrix for a given list of n-values
    for the dimension d = 1, 2, 3. N = (n-1)^d is the dimension of the Hilbert-matrix.

    Parameters
    ----------
    n_list (list of ints): The n-values for which to plot the condition.
    d (int): dimension of the space
    """
    conditions = []
    for n in n_list:    #pylint: disable=invalid-name
        h = sc.linalg.hilbert((n-1)**d) #pylint: disable=invalid-name
        conditions.append(get_cond_matrix(h))
    print(conditions) #pylint: disable=superfluous-parens


def plot_cond_hilbert(n_list, d): #pylint: disable=invalid-name
    """
    Plots the condition of the Hilbert-matrix for a given list of n-values
    for the dimension d = 1, 2, 3. N = (n-1)^d is the dimension of the Hilbert-matrix.

    Parameters
    ----------
    n_list (list of ints): The n-values for which to plot the condition.
    d (int): dimension of the space
    """
    numbers_of_points = []
    conditions = []
    for n in n_list: #pylint: disable=invalid-name
        h = sc.linalg.hilbert((n-1)**d) #pylint: disable=invalid-name
        conditions.append(get_cond_matrix(h))
        numbers_of_points.append((n-1)**d)
    plt.loglog(numbers_of_points, conditions, "mo")
    plt.xlabel('$N$')
    plt.ylabel('condition of $H_N$')
    plt.title('Condition of $H_N$ for d=' + str(d) + ' ($N=(n-1)^d$)')
    plt.grid()
    plt.show()


def main():
    """ Main function to demonstrate the functionality of our modules.
    """

# # Fehlerentwicklung pro Iteration

    rhs.plot_iterates_error(u1, f1, 1, 100)
    rhs.plot_iterates_error(u2, f2, 2, 100)
    rhs.plot_iterates_error(u3, f3, 3, 100)

# # Fehler-/Konvergenzplot Vergleich mit LU

    rhs.plot_error_comp(u1, f1, 1, np.geomspace(3, 10000, num=10, dtype=int))
    rhs.plot_error_comp(u2, f2, 2, np.geomspace(3, 100, num=10, dtype=int))
    rhs.plot_error_comp(u3, f3, 3, np.geomspace(3, 24, num=10, dtype=int))

# # Fehlerentwicklung für verschiedene Epsilon

    rhs.plot_error_eps(u1, f1, 1, np.geomspace(3, 10000, num=10, dtype=int))
    rhs.plot_error_eps(u2, f2, 2, np.geomspace(3, 100, num=10, dtype=int))
    rhs.plot_error_eps(u3, f3, 3, np.geomspace(3, 24, num=10, dtype=int))

# # # # Fehlerentwicklung und Kondition der Matrix
#
#     rhs.plot_error_cond(u1, f1, 1, np.geomspace(3, 10000, num=10, dtype=int))
#     rhs.plot_error_cond(u2, f2, 2, np.geomspace(3, 100, num=10, dtype=int))
#     rhs.plot_error_cond(u3, f3, 3, np.geomspace(3, 24, num=10, dtype=int))



if __name__ == "__main__":
    main()
