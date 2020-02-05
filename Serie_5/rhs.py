"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2020_01_03

Implements the right hand side vector of the system to solve the Poisson-problem.
Also calculates and plots the error of the numerical solution of the Poisson-problem
with respect to the row sum norm.
"""
# pylint: disable=invalid-name, no-member, import-error, wrong-import-position

from matplotlib import use
# use('qt4agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d # pylint: disable=unused-import
import numpy as np
import scipy.linalg as lina
import block_matrix
import linear_solvers

plt.rcParams['font.size'] = 12

def rhs(d, n, f): #pylint: disable=invalid-name
    """ Computes the right-hand side vector b for a given function f.
    Parameters
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is f(x).
        Here x is a scalar or array_like of numpy. The return value is a scalar.
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

def compute_error(d, n, hat_u, u): #pylint: disable=invalid-name
    """ Computes the error of the numerical solution of the Poisson-problem
    with respect to the max norm.

    Parameters
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intersections in each dimension
    hat_u : array_like of numpy
            Finite difference approximation of the solution of the Poisson-problem
            at the discretization points
    u : callable
        Solution of the Poisson-problem
        The calling signature is u(x). Here x is a scalar
        or array_like of numpy. The return value is a scalar.

    Returns
    -------
    float
       maximal absolute error at the disretization points
    """
    actual_u = rhs(d, n, u)*(n**2)
    err = [abs(actual_u[i]-hat_u[i]) for i in range(len(actual_u))]
    return max(err)

def plot_error(u, f, d, n_list): #pylint: disable=invalid-name
    """ Plots the maxima of absolute errors of the numerical solution of the Poisson-problem
    for a given list of n-values. N = (n-1)^d is the dimension of the block matrix.

    Parameters
    ----------
    u : callable
        Solution of the Poisson-problem
        The calling signature is u(x). Here x is a scalar
        or array_like of numpy. The return value is a scalar.
    f : callable
        Input function of the Poisson-problem
        The calling signature is f(x). Here x is a scalar
        or array_like of numpy. The return value is a scalar.
    d: int
        Dimension of the Poisson-problem
    n_list: list of ints
        The n-values for which to plot the errors.
    """
    numbers_of_points = []
    errors = []
    for n in n_list:
        A = block_matrix.BlockMatrix(d, n)
        b = rhs(d, n, f)
        lu = A.get_lu()
        hat_u = linear_solvers.solve_lu(lu[0], lu[1], lu[2], lu[3], b)

        errors.append(compute_error(d, n, hat_u, u))
        numbers_of_points.append((n-1)**d)

    numbers_of_points_pow1 = [np.float_(N)**(-1) for N in numbers_of_points]
    numbers_of_points_pow2 = [np.float_(N)**(-2) for N in numbers_of_points]
    numbers_of_points_pow3 = [np.float_(N)**(-1/2) for N in numbers_of_points]

    plt.loglog(numbers_of_points, numbers_of_points_pow3, label='$N^{-1/2}$',
               color='lightgray', linestyle=':')
    plt.loglog(numbers_of_points, numbers_of_points_pow1, label='$N^{-1}$',
               color='lightgray')
    plt.loglog(numbers_of_points, numbers_of_points_pow2, label='$N^{-2}$',
               color='lightgray', linestyle='-.')

    plt.loglog(numbers_of_points, errors, 'go--')
    plt.xlabel('$N$')
    plt.ylabel('maximum of absolute error')
    plt.title('Maxima of absolute errors for $d$ = ' + str(d))
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure()

def plot_error_comp(u, f, d, n_list): #pylint: disable=invalid-name
    """ Plots the maxima of absolute errors of the numerical solution of the Poisson-problem
    for a given list of n-values. N = (n-1)^d is the dimension of the block matrix.

    Parameters
    ----------
    u : callable
        Solution of the Poisson-problem
        The calling signature is u(x). Here x is a scalar
        or array_like of numpy. The return value is a scalar.
    f : callable
        Input function of the Poisson-problem
        The calling signature is f(x). Here x is a scalar
        or array_like of numpy. The return value is a scalar.
    d: int
        Dimension of the Poisson-problem
    n_list: list of ints
        The n-values for which to plot the errors.
    """
    numbers_of_points = []
    errors_cg = []
    errors_lu = []
    for n in n_list:
        A = block_matrix.BlockMatrix(d, n)
        b = rhs(d, n, f)

        lu = A.get_lu()
        hat_u = linear_solvers.solve_lu(lu[0], lu[1], lu[2], lu[3], b)

        errors_lu.append(compute_error(d, n, hat_u, u))

        cg = linear_solvers.solve_cg(A.get_sparse(), b, np.zeros((n-1)**d),
                                     params=dict(eps=1e-8, max_iter=(2*(n-1)**2*d), min_red=0))
        print(cg[0])
        errors_cg.append(compute_error(d, n, cg[1][-1], u))
        numbers_of_points.append((n-1)**d)

    # numbers_of_points_pow1 = [np.float_(N)**(-1) for N in numbers_of_points]
    # numbers_of_points_pow2 = [np.float_(N)**(-2) for N in numbers_of_points]
    # numbers_of_points_pow3 = [np.float_(N)**(-1/2) for N in numbers_of_points]
    #
    # plt.loglog(numbers_of_points, numbers_of_points_pow3, label='$N^{-1/2}$',
    #            color='lightgray', linestyle=':')
    # plt.loglog(numbers_of_points, numbers_of_points_pow1, label='$N^{-1}$',
    #            color='lightgray')
    # plt.loglog(numbers_of_points, numbers_of_points_pow2, label='$N^{-2}$',
    #            color='lightgray', linestyle='-.')

    plt.loglog(numbers_of_points, errors_cg, 'gX-', label='CG')
    plt.loglog(numbers_of_points, errors_lu, 'r.--', label='LU')
    plt.xlabel('$N$')
    plt.ylabel('maximum of absolute error')
    plt.title('Maxima of absolute errors for $d$ = ' + str(d))
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure()

def plot_error_list_comp(u_list, f_list, n_list_list): #pylint: disable=invalid-name, too-many-locals
    """ Plots the maxima of absolute errors of the numerical solution of the Poisson-problem
    for a given list of n-values and for the dimension d = 1, 2, 3.

    Parameters
    ----------
    n_list_list: list of list of ints
        The n-values for which to plot the errors.
    u_list : list of callable functions
        Solution of the Poisson-problem
        The calling signature is u(x). Here x is a scalar
        or array_like of numpy. The return value is a scalar.
    f_list : list of callable functions
        Input function of the Poisson-problem
        The calling signature is f(x). Here x is a scalar
        or array_like of numpy. The return value is a scalar.
    """

    numbers_of_points_1 = []
    errors_1 = []
    errors_cg1 = []
    for n in n_list_list[0]:
        A = block_matrix.BlockMatrix(1, n)
        b = rhs(1, n, f_list[0])
        lu = A.get_lu()
        hat_u = linear_solvers.solve_lu(lu[0], lu[1], lu[2], lu[3], b)

        errors_1.append(compute_error(1, n, hat_u, u_list[0]))

        cg = linear_solvers.solve_cg(A.get_sparse(), b, np.zeros((n-1)),
                                     params=dict(eps=1e-8, max_iter=(2*(n-1)**2), min_red=0))
        print(cg[0])
        errors_cg1.append(compute_error(1, n, cg[1][-1], u_list[0]))

        numbers_of_points_1.append((n-1)**1)

    numbers_of_points_2 = []
    errors_2 = []
    errors_cg2 = []
    for n in n_list_list[1]:
        A = block_matrix.BlockMatrix(2, n)
        b = rhs(2, n, f_list[1])
        lu = A.get_lu()
        hat_u = linear_solvers.solve_lu(lu[0], lu[1], lu[2], lu[3], b)

        cg = linear_solvers.solve_cg(A.get_sparse(), b, np.zeros((n-1)**2),
                                     params=dict(eps=1e-8, max_iter=(2*(n-1)**2*2), min_red=0))
        print(cg[0])
        errors_cg2.append(compute_error(2, n, cg[1][-1], u_list[1]))

        errors_2.append(compute_error(2, n, hat_u, u_list[1]))
        numbers_of_points_2.append((n-1)**2)

    numbers_of_points_3 = []
    errors_3 = []
    errors_cg3 = []
    for n in n_list_list[2]:
        A = block_matrix.BlockMatrix(3, n)
        b = rhs(3, n, f_list[2])
        lu = A.get_lu()
        hat_u = linear_solvers.solve_lu(lu[0], lu[1], lu[2], lu[3], b)

        cg = linear_solvers.solve_cg(A.get_sparse(), b, np.zeros((n-1)**3),
                                     params=dict(eps=1e-8, max_iter=(2*(n-1)**2*3), min_red=0))
        print(cg[0])
        errors_cg3.append(compute_error(3, n, cg[1][-1], u_list[2]))

        errors_3.append(compute_error(3, n, hat_u, u_list[2]))
        numbers_of_points_3.append((n-1)**3)

    numbers_of_points_pow1 = [np.float_(N)**(-1) for N in numbers_of_points_3]
    numbers_of_points_pow2 = [np.float_(N)**(-2) for N in numbers_of_points_3]
    numbers_of_points_pow3 = [np.float_(N)**(-1/2) for N in numbers_of_points_3]

    plt.loglog(numbers_of_points_3, numbers_of_points_pow3, label='$N^{-1/2}$',
               color='lightgray')
    plt.loglog(numbers_of_points_3, numbers_of_points_pow1, label='$N^{-1}$',
               color='lightgray', linestyle='-.')
    plt.loglog(numbers_of_points_3, numbers_of_points_pow2, label='$N^{-2}$',
               color='lightgray', linestyle=':')

    plt.loglog(numbers_of_points_1, errors_1, label='LU $d=1$', linestyle='-',
               color='cornflowerblue', marker='X')
    plt.loglog(numbers_of_points_2, errors_2, label='LU $d=2$', linestyle='-',
               color='blue', marker='X')
    plt.loglog(numbers_of_points_3, errors_3, label='LU $d=3$', linestyle='-',
               color='navy', marker='X')

    plt.loglog(numbers_of_points_1, errors_cg1, label='CG $d=1$', linestyle='-.',
               color='tomato', marker='.')
    plt.loglog(numbers_of_points_2, errors_cg2, label='CG $d=2$', linestyle='-.',
               color='red', marker='.')
    plt.loglog(numbers_of_points_3, errors_cg3, label='CG $d=3$', linestyle='-.',
               color='darkred', marker='.')

    plt.xlabel('$N$')
    plt.ylabel('maximum of absolute error')
    plt.legend()
    plt.title('Maxima of absolute errors for $d=1,2,3$')
    plt.grid()
    plt.show()

def plot_error_eps(u, f, d, n_list):
    k_list = [-2, 0, 2, 4, 6]
    markers = ['s','X' , 'H', '^', '.']
    i = 0
    for k in k_list:
        numbers_of_points = []
        errors_cg = []
        for n in n_list:
            A = block_matrix.BlockMatrix(d, n)
            b = rhs(d, n, f)

            cg = linear_solvers.solve_cg(A.get_sparse(), b, np.zeros((n-1)**d),
                                         params=dict(eps=float(n)**(-k), max_iter=(2*(n-1)**2*d), min_red=0))
            print(cg[0])
            errors_cg.append(compute_error(d, n, cg[1][-1], u))
            numbers_of_points.append((n-1)**d)
        if i == 0:
            if d == 1:
                conv0 = [(10*float(N))**(1/8) for N in numbers_of_points]
                conv2 = [float(N)**(-2) for N in numbers_of_points]

                plt.loglog(numbers_of_points, conv0, label='$(10N)^{1/8}$',
                           color='lightgray')
                plt.loglog(numbers_of_points, conv2, label='$N^{-2}$',
                           color='lightgray', linestyle='-.')

            elif d == 2:
                conv0 = [(10*float(N))**(1/8) for N in numbers_of_points]
                conv1 = [(10*float(N))**(-1/2) for N in numbers_of_points]
                conv2 = [(10*float(N))**(-1) for N in numbers_of_points]

                plt.loglog(numbers_of_points, conv0, label='$(10N)^{1/8}$',
                           color='lightgray')
                plt.loglog(numbers_of_points, conv1, label='$(10N)^{-1/2}$',
                           color='lightgray', linestyle='--')
                plt.loglog(numbers_of_points, conv2, label='$(10N)^{-1}$',
                           color='lightgray', linestyle='-.')
            else:
                conv0 = [(0.000000001*float(N))**(1/8) for N in numbers_of_points]
                conv1 = [(1000*float(N))**(-1/2) for N in numbers_of_points]
                conv2 = [(10000*float(N))**(-1/4) for N in numbers_of_points]

                plt.loglog(numbers_of_points, conv0, label='$(10^{-8}N)^{1/8}$',
                           color='lightgray')
                plt.loglog(numbers_of_points, conv2, label='$(1000N)^{-1/4}$',
                           color='lightgray', linestyle='--')
                plt.loglog(numbers_of_points, conv1, label='$(10000N)^{-1/2}$',
                           color='lightgray', linestyle='-.')


        plt.loglog(numbers_of_points, errors_cg, '--', marker=markers[i], label='k='+str(k))
        i = i+1

    plt.xlabel('$N$')
    plt.ylabel('maximum of absolute error')
    plt.title('Maxima of absolute errors for $d$ = ' + str(d))
    plt.legend(loc='lower left')
    plt.grid()
    plt.show()
    plt.figure()

def plot_error_cond(u, f, d, n_list):
    numbers_of_points = []
    errors_cg = []
    conditions = []
    for n in n_list:
        A = block_matrix.BlockMatrix(d, n)
        b = rhs(d, n, f)

        conditions.append(A.get_cond())

        cg = linear_solvers.solve_cg(A.get_sparse(), b, np.zeros((n-1)**d),
                                     params=dict(eps=1e-8, max_iter=(2*(n-1)**2*d), min_red=0))
        print(cg[0])
        errors_cg.append(lina.norm(cg[2][-1], np.inf))
        numbers_of_points.append((n-1)**d)

    plt.loglog(numbers_of_points, conditions, "mo", label='condition')
    plt.loglog(numbers_of_points, errors_cg, 'go--', label='CG')
    plt.xlabel('$N$')
    plt.ylabel('maximum of absolute error')
    plt.title('Maxima of absolute errors for $d$ = ' + str(d))
    plt.legend()
    plt.grid()
    plt.show()
    plt.figure()

def plot_iterates_error(u, f, d, n,
             params=dict(eps=1e-8, max_iter=1000, min_red=1e-4)):
    A = block_matrix.BlockMatrix(d, n).get_sparse()
    b = rhs(d, n, f)
    cg = linear_solvers.solve_cg(A, b, np.zeros((n-1)**d), params)
    print(cg[0])

    # errors = [lina.norm(r, np.inf) for r in cg[2]]
    errors = [compute_error(d, n, sol, u) for sol in cg[1]]

    plt.plot(range(len(errors)), errors, 'm.--')
    # plt.yscale('log')
    plt.xlabel('number of iteration')
    plt.ylabel('absolute error')
    plt.title('Development of absolute error for d = %d and n = %d' %(d, n))
    plt.grid()
    plt.show()
    plt.figure()

def plot_error_list(u_list, f_list, n_list_list): #pylint: disable=invalid-name, too-many-locals
    """ Plots the maxima of absolute errors of the numerical solution of the Poisson-problem
    for a given list of n-values and for the dimension d = 1, 2, 3.

    Parameters
    ----------
    n_list_list: list of list of ints
        The n-values for which to plot the errors.
    u_list : list of callable functions
        Solution of the Poisson-problem
        The calling signature is u(x). Here x is a scalar
        or array_like of numpy. The return value is a scalar.
    f_list : list of callable functions
        Input function of the Poisson-problem
        The calling signature is f(x). Here x is a scalar
        or array_like of numpy. The return value is a scalar.
    """

    numbers_of_points_1 = []
    errors_1 = []
    for n in n_list_list[0]:
        A = block_matrix.BlockMatrix(1, n)
        b = rhs(1, n, f_list[0])
        lu = A.get_lu()
        hat_u = linear_solvers.solve_lu(lu[0], lu[1], lu[2], lu[3], b)

        errors_1.append(compute_error(1, n, hat_u, u_list[0]))
        numbers_of_points_1.append((n-1)**1)

    numbers_of_points_2 = []
    errors_2 = []
    for n in n_list_list[1]:
        A = block_matrix.BlockMatrix(2, n)
        b = rhs(2, n, f_list[1])
        lu = A.get_lu()
        hat_u = linear_solvers.solve_lu(lu[0], lu[1], lu[2], lu[3], b)

        errors_2.append(compute_error(2, n, hat_u, u_list[1]))
        numbers_of_points_2.append((n-1)**2)

    numbers_of_points_3 = []
    errors_3 = []
    for n in n_list_list[2]:
        A = block_matrix.BlockMatrix(3, n)
        b = rhs(3, n, f_list[2])
        lu = A.get_lu()
        hat_u = linear_solvers.solve_lu(lu[0], lu[1], lu[2], lu[3], b)

        errors_3.append(compute_error(3, n, hat_u, u_list[2]))
        numbers_of_points_3.append((n-1)**3)

    numbers_of_points_pow1 = [np.float_(N)**(-1) for N in numbers_of_points_3]
    numbers_of_points_pow2 = [np.float_(N)**(-2) for N in numbers_of_points_3]
    numbers_of_points_pow3 = [np.float_(N)**(-1/2) for N in numbers_of_points_3]

    plt.loglog(numbers_of_points_3, numbers_of_points_pow3, label='$N^{-1/2}$',
               color='lightgray')
    plt.loglog(numbers_of_points_3, numbers_of_points_pow1, label='$N^{-1}$',
               color='lightgray', linestyle='-.')
    plt.loglog(numbers_of_points_3, numbers_of_points_pow2, label='$N^{-2}$',
               color='lightgray', linestyle=':')

    plt.loglog(numbers_of_points_1, errors_1, label='$d=1$', linestyle='--',
               color='blue')
    plt.loglog(numbers_of_points_2, errors_2, label='$d=2$', linestyle='--',
               color='magenta')
    plt.loglog(numbers_of_points_3, errors_3, label='$d=3$', linestyle='--',
               color='red')

    plt.xlabel('$N$')
    plt.ylabel('maximum of absolute error')
    plt.legend()
    plt.title('Maxima of absolute errors for $d=1,2,3$')
    plt.grid()
    plt.show()


def plot_functions(u, f, n): #pylint: disable=invalid-name, too-many-locals
    """ Plots the numerical, the exact solution of our Poisson-problem (dimension d=2)
    for a given value of n (n is the number of intersections in each dimension
    and their absolute and their relative difference.

    Parameters
    ----------
    u : callable
        Solution of the Poisson-problem
        The calling signature is u(x). Here x is a scalar
        or array_like of numpy. The return value is a scalar.
    f : callable
        Input function of the Poisson-problem
        The calling signature is f(x). Here x is a scalar
        or array_like of numpy. The return value is a scalar.
    n:  int
        The n-value for which to plot the functions.
    """
    x = np.linspace(0, 1, n+1)
    y = np.linspace(0, 1, n+1)
    X, Y = np.meshgrid(x, y)

    exact = u((X, Y))

    A = block_matrix.BlockMatrix(2, n)
    b = rhs(2, n, f)
    lu = A.get_lu()
    hat_u = linear_solvers.solve_lu(lu[0], lu[1], lu[2], lu[3], b)
    approx = np.reshape(hat_u, (-1, n-1))

    vzeroes = []
    for _ in range(n-1):
        vzeroes.append([0])
    hzeroes = np.zeros(n+1)

    approx = np.hstack((vzeroes, approx, vzeroes))
    approx = np.vstack((hzeroes, approx, hzeroes))

    fig = plt.figure()
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    #ax4 = fig.add_subplot(224, projection='3d')

    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_zlabel('$\hat{u}(x)$') #pylint: disable=anomalous-backslash-in-string
    ax1.plot_surface(X, Y, approx, cmap='viridis', edgecolor='none')
    ax1.set_title('Approximate solution')

    difference = abs(approx - exact)
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_zlabel('$|\hat{u}(x)-u(x)|$') #pylint: disable=anomalous-backslash-in-string
    ax2.plot_surface(X, Y, difference, cmap='viridis', edgecolor='none')
    ax2.set_title('Difference')

    ax3.set_xlabel('$x_1$')
    ax3.set_ylabel('$x_2$')
    ax3.set_zlabel('$u(x)$')
    ax3.plot_surface(X, Y, exact, cmap='viridis', edgecolor='none')
    ax3.set_title('Exact solution')

    #rel_difference = difference/(exact+np.mean(difference))
    #ax4.plot_surface(X, Y, rel_difference, cmap='viridis', edgecolor='none')
    #ax4.set_title('Relative Difference')

    plt.show()
