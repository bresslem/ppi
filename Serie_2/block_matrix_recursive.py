"""
Author: Bressler_Marisa, Jeschke_Anne
Date:
"""

import scipy as sp
import numpy as np

class BlockMatrix:
    """ Represents block matrices arising from finite difference approximations
    of the Laplace operator.

    Parameters
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intervals in each dimension

    Attributes
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intervals in each dimension
    """

    def __init__(self, d, n):
        self.d = d
        self.n = n

    def get_sparse(self):
        """ Returns the block matrix as sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            block_matrix in a sparse data format
        """


    def get_A_l(self, l):
        """ Returns the matrix of index l as sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            block_matrix in a sparse data format
        """
        if (l == 1):
            dk = np.array([-np.ones(self.n-2), np.full((self.n-1), 2*self.d), -np.ones(self.n-2)])
            offset = [-1,0,1]
            A_1 = diags(k,offset).toarray()
            return csr_matrix(A_1)
        else:
            A_prev = self.get_A_l(l-1)
            I_l = sp.sparse.identity(l, format='csr')
            A_l = sp.sparse.hstack([A_l,I_l], format = 'csr'))
            # use hstack/vstack to create marix

            return A_l

    def eval_zeros(self):
        """ Returns the (absolute and relative) numbers of (non-)zero elements
        of the matrix. The relative number of the (non-)zero elements are with
        respect to the total number of elements of the matrix.

        Returns
        -------
        int
            number of non-zeros
        int
            number of zeros
        float
            relative number of non-zeros
        float
            relative number of zeros
        """


def main():
    A_d = BlockMatrix(2, 4)

if __name__ == "__main__":
    main()
