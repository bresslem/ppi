"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2020_01_17

Main program to demonstrate the functionality of our modules.
"""

import numpy as np
import scipy as sc
import scipy.linalg as lina

def get_qr(A):
    """
    Function to get the QR decomposition of the input matrix.

    Parameters
    ----------
    A: array_like
        Matrix of which to get the QR decomposition.
    Returns
    -------
    numpy.ndarray
        Matrix Q of the QR decomposition.
    numpy.ndarray
        Matrix R of the QR decomposition.
    """
    return lina.qr(A)

def full_rank(A):
    """
    Function to test if A has a full column rank.

    Parameters
    ----------
    A: array_like
        Matrix to test.
    Returns
    -------
    boolean
        true if A has full column rank.
    """
    R = get_qr(A)

    for i in range(R.shape[1]):
        if R[i,i] == 0:
            return false

    return true

def solve_qr(A, b):
    """
    Function to solve Ax = b for given A and b.

    Parameters
    ----------
    A: array_like
    b: array_like

    Returns
    -------
    np.ndarray
        solution x of Ax=b.

    Raises
    -------
    Error:
        If A does not have a full rank.
    """

    if not full_rank(A):
        raise Error('A does not have a full column rank.')

    qr = get_qr(A)
    z = np.dot(sc.transpose(qr[0]), b).resize(A.shape[1])
    r = qr[1].resize(A.shape[1])
    return lina.solve_triangular(r, z)

def norm_of_residuum(A, b):
    """
    Function to get the norm of Ax-b

    Parameters
    ----------
    A: array_like
    b: array_like

    Returns
    -------
    float
        euklid norm of Ax-b

    Raises
    -------
    Error:
        If A does not have a full rank.
    """

    if not full_rank(A):
        raise Error('A does not have a full column rank.')

    z = np.dot(sc.transpose(get_qr(A)[0]), b).resize(A.shape[0]-A.shape[1])
    return lina.norm(z, 2)

def main():
    """ Main function to demonstrate the functionality of our modules.
    """


if __name__ == "__main__":
    main()
