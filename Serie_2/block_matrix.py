"""
Author: Bressler_Marisa, Jeschke_Anne
Date:
"""

import scipy as sp
import numoy as np

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
            data = np.array([-2*self.d,-1])

            for i in range(n-3):
                data = np.append(data, [-1, 2*self.d, -1])

            data = np.append(data, [-1, 2*self.d])

            return

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