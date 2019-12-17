"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2019_12_04

Main program to demonstrate the functionality of our modules.
"""
import block_matrix
import rhs
import numpy as np
import scipy as sc
import linear_solvers
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


def get_cond_matrix(a):
        """ Computes the condition number of the matrix a.

        Returns
        -------
        float
            condition number with respect to the row sum norm
        """
        return (np.linalg.norm(a, np.inf)*np.linalg.norm(np.linalg.inv(a), np.inf))


def print_cond_hilbert(n_list, d):
    """
    Calculates the condition of the Hilbert-matrix for a given list of n-values
    for the dimension d = 1, 2, 3. N = (n-1)^d is the dimension of the Hilbert-matrix.

    Parameters
    ----------
    n_list (list of ints): The n-values for which to plot the condition.
    d (int): dimension of the space
    """
    conditions = []
    for n in n_list:
        h = sc.linalg.hilbert((n-1)**d)
        conditions.append(get_cond_matrix(h))
    print(conditions)


def plot_cond_hilbert(n_list, d):
    """
    Plots the condition of the Hilbert-matrix for a given list of n-values
    for the dimension d = 1, 2, 3. N = (n-1)^d is the dimension of the Hilbert-matrix.

    Parameters
    ----------
    n_list (list of ints): The n-values for which to plot the condition.
    d (int): dimension of the space
    """
    #pylint: disable=invalid-name
    numbers_of_points = []
    conditions = []
    for n in n_list: #pylint: disable=invalid-name
        h = sc.linalg.hilbert((n-1)**d)
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
    rhs.plot_error(u1, f1, 1, np.geomspace(2, 10000, num=10, dtype=int))
    rhs.plot_error(u2, f2, 2, np.geomspace(2, 100, num=10, dtype=int))
    rhs.plot_error(u3, f3, 3, np.geomspace(2, 10, num=10, dtype=int))

    # np.geomspace(2, 100, num=5, dtype=int)
    # list = range(2, 26)
    # print(list)
    # rhs.plot_error_list([u1, u2, u3], [f1, f2, f3], list)
    #
    # n = 10
    # A = block_matrix.BlockMatrix(3, n)
    # print(1)
    # b = rhs.rhs(3, n, f3)
    # print(2)
    # lu = A.get_lu()
    # print(3)
    # hat_u = linear_solvers.solve_lu(lu[0], lu[1], lu[2], lu[3], b)
    # print(rhs.compute_error(3, 37, hat_u, u3))
    #
    # block_matrix.plot_cond(range(2, 11), 1)
    # plot_cond_hilbert(range(2, 11), 1)

    # block_matrix.print_cond(range(2, 11), 1)
    # print_cond_hilbert(range(2, 11), 1)


    # block_matrix.plot_cond(range(2, 26), 2)
    # plot_cond_hilbert(range(2, 26), 2)

#    block_matrix.print_cond(range(2, 11), 2)
#    print_cond_hilbert(range(2, 11), 2)


#    block_matrix.plot_cond(range(2, 11), 3)
#    plot_cond_hilbert(range(2, 11), 3)

#    block_matrix.print_cond(range(2, 11), 3)
#    print_cond_hilbert(range(2, 11), 3)



#    block_matrix.plot_non_zeros(range(2, 11))



    # rhs.plot_functions(u2, f2, 5)
    # rhs.plot_functions(u2, f2, 10)
    # rhs.plot_functions(u2, f2, 25)
    # rhs.plot_functions(u2, f2, 50)

if __name__ == "__main__":
    main()
