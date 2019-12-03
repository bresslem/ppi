"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2019_11_26

Module to implement the right hand side vector of the system to solve the Poisson-Problem.
"""
# pylint: disable=invalid-name
import numpy as np

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
    """ Computes the error of the numerical solution of the Poisson problem
    with respect to the max-norm.

    Parameters
    ----------
    d : int
       Dimension of the space
    n : int
        Number of intersections in each dimension
    hat_u : array_like of ’numpy’
       Finite difference approximation of the solution of the Poisson problem
       at the discretization points
    u : callable
        Solution of the Possion problem
        The calling signature is ’u(x)’. Here ’x’ is a scalar
        or array_like of ’numpy’. The return value is a scalar.

    Returns
    -------
    float
       maximal absolute error at the disretization points
    """
    actual_u = rhs(d, n, u)/((1/n)**2)
    err = [abs(actual_u[i]-hat_u[i]) for i in range(len(actual_u)+1)]
    return max(err)

def plot_error(n_array):
    """ Plots errors of solution of Poisson Problem for a given array of
    Ns.

    Parameters
    ----------
    n_array: list
        List of N
    """