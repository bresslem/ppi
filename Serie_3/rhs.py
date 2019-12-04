"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2019_12_04

Implements the right hand side vector of the system to solve the Poisson-problem.
Also calculates and plots the error of the numerical solution of the Poisson-problem
with respect to the row sum norm.
"""
# pylint: disable=invalid-name
import numpy as np
import block_matrix
import linear_solvers
from matplotlib import use
#use('qt4agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12

def rhs(d, n, f):
    """ Computes the right-hand side vector `b` for a given function `f`.
    Parameters
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is `f(x)`.
        Here `x` is a scalar or array_like of `numpy`. The return value is a scalar.
    Returns
    -------
    numpy.ndarray
        Vector to the right-hand-side f.
    Raises
    ------
    ValueError
        If d < 1 or n < 2.
    """
    if d < 1 or n < 2:
        raise ValueError('d cannot be smaller than 1 and n cannot be smaller than 2.')

    rhs_vector = np.zeros((n-1)**d)

    if d == 1:
        for i in range(1, n):
            rhs_vector[i-1] = ((1/n)**2)*f((i/n))
    else:
        for i in range(1, ((n-1)**d)+1):
            x = np.zeros(d)
            s = i
            for l in range(d, 1, -1):
                if s%((n-1)**(l-1)) == 0:
                    x[l-1] = (s//((n-1)**(l-1)))/n
                    s = (n-1)**(l-1)
                else:
                    x[l-1] = (s//((n-1)**(l-1))+1)/n
                    s = s%((n-1)**(l-1))
                x[0] = s/n
            rhs_vector[i-1] = ((1/n)**2)*f(x)
    return rhs_vector

def compute_error(d, n, hat_u, u):
    """ Computes the error of the numerical solution of the Poisson-problem
    with respect to the max norm.

    Parameters
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intersections in each dimension
    hat_u : array_like of ’numpy’
            Finite difference approximation of the solution of the Poisson-problem
            at the discretization points
    u : callable
        Solution of the Poisson-problem
        The calling signature is ’u(x)’. Here ’x’ is a scalar
        or array_like of ’numpy’. The return value is a scalar.

    Returns
    -------
    float
       maximal absolute error at the disretization points
    """
    actual_u = rhs(d, n, u)*(n**2)
    err = [abs(actual_u[i]-hat_u[i]) for i in range(len(actual_u))]
    return max(err)


def plot_error(u, f, d, n_array):
    """ Plots the maxima of absolute errors of the numerical solution of the Poisson-problem
    for a given array of n-values. N = (n-1)^d is the dimension of the block matrix.

    Parameters
    ----------
    n_array: list of ints
             The n-values for which to plot the errors.
    u : callable
        Solution of the Poisson-problem
        The calling signature is ’u(x)’. Here ’x’ is a scalar
        or array_like of ’numpy’. The return value is a scalar.
    f : callable
        Input function of the Poisson-problem
        The calling signature is ’f(x)’. Here ’x’ is a scalar
        or array_like of ’numpy’. The return value is a scalar.
    """
    numbers_of_points = []
    errors = []
    for n in n_array:
        A = block_matrix.BlockMatrix(d, n)
        b = rhs(d, n, f)
        lu = A.get_lu()
        hat_u = linear_solvers.solve_lu(lu[0], lu[1], lu[2], lu[3], b)

        errors.append(compute_error(d, n, hat_u, u))
        numbers_of_points.append((n-1)**d)
    plt.plot(numbers_of_points, errors, "go--")
    plt.xlabel('$N$')
    plt.ylabel('maximum of absolute error')
    plt.title('Maxima of absolute errors for d = ' + str(d))
    plt.grid()
    plt.show()
    plt.figure()
