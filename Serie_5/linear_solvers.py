"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2020_01_03

Solves the given linear equation system Ax = b via forward and backward substitution
given the LU-decomposition with full pivoting pr * A * pc = l * u and with CG method.
"""

#pylint: disable=invalid-name, dangerous-default-value, import-error, unused-import

import scipy.linalg as lina
import scipy.sparse
import numpy as np

def solve_lu(pr, l, u, pc, b):
    """ Solves the linear system Ax = b via forward and backward substitution
    given the decomposition pr * A * pc = l * u.

    Parameters
    ----------
    pr : scipy.sparse.csr_matrix
         row permutation matrix of LU-decomposition
    l : scipy.sparse.csr_matrix
        lower triangular unit diagonal matrix of LU-decomposition
    u : scipy.sparse.csr_matrix
        upper triangular matrix of LU-decomposition
    pc : scipy.sparse.csr_matrix
         column permutation matrix of LU-decomposition
    b : numpy.ndarray
        vector of the right-hand-side of the linear system

    Returns
    -------
    x : numpy.ndarray
       solution of the linear system
    """

    z = lina.solve_triangular(l.toarray(), np.dot(pr.toarray(), b),
                              lower=True, unit_diagonal=True)
    y = lina.solve_triangular(u.toarray(), z)
    x = np.dot(pc.toarray(), y)
    return x

def solve_cg(A, b, x0,
             params=dict(eps=1e-8, max_iter=1000, min_red=1e-4)):
    """ Solves the linear system Ax = b via the conjugated gradient method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray
        right-hand-side of the linear system
    x0 : numpy.ndarray
        initial guess of the solution

    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm
        max_iter : int
            maximal number of iterations that the solver will perform.
            If set less or equal to 0 no constraint on the number of iterations is imposed.
        min_red : float
            minimal reduction of the residual in each step

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list
        iterates of the algorithm. First entry is `x0`.
    list
        residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., `eps=0` and `max_iter=0`,
        etc.
    """
    if params["max_iter"] <= 0 and params["min_red"] <= 0 and params["eps"] <= 0:
        raise ValueError('No termination condition provided.')

    iterates = [x0]
    residuals = [A.dot(x0) - b]

    d = -residuals[0]

    for k in range(params["max_iter"]+1):
        z = A.dot(d)

        c = np.dot(residuals[k], residuals[k])/np.dot(d, z)

        iterates.append(iterates[k] + c*d)
        residuals.append(residuals[k] + c*z)

        if abs(lina.norm(residuals[k+1], np.inf)
               - lina.norm(residuals[k], np.inf)) < params["min_red"]:
            return "min_red", iterates, residuals

        if lina.norm(residuals[k+1], np.inf) < params["eps"]:
            return "eps", iterates, residuals

        beta = (np.dot(residuals[k+1], residuals[k+1])/np.dot(residuals[k], residuals[k]))
        d = -residuals[k+1] + beta*d

    return "max_iter", iterates, residuals
