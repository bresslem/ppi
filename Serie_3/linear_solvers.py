"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2019_12_04

Solves the given linear equation system Ax = b via forward and backward substitution
given the LU-decomposition with full pivoting pr * A * pc = l * u.
"""
import scipy.linalg as lina
import numpy as np
import block_matrix

def solve_lu(pr, l, u, pc, b): #pylint: disable=invalid-name
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

    z = lina.solve_triangular(l.toarray(), np.dot(pr.toarray(),b), lower=True, unit_diagonal=True)
    y = lina.solve_triangular(u.toarray(), z)
    x = np.dot(pc.toarray(), y)
    return x
