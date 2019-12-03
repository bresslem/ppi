import scipy.linalg as lina
import numpy as np
import block_matrix

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

    Returns
    -------
    x : numpy.ndarray
       solution of the linear system
    """

    y = lina.solve_triangular(l, np.matmul(np.matmul(pr, b),pc),lower=True, unit_diagonal=True)

    return lina.solve_triangular(u, y)