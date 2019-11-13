"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2019_11_13
"""
# pylint: disable=invalid-name
import scipy.sparse as sps
import numpy as np

class BlockMatrix:
    """ Represents block matrices arising from finite difference approximations
    of the Laplace operator.

    Parameters
    ----------
    d (int): Dimension of the space
    n (int): Number of intervals in each dimension

    Attributes
    ----------
    d (int): Dimension of the space
    n (int): Number of intervals in each dimension
    """

    def __init__(self, d, n):
        self.d = d
        self.n = n

    def get_A_l(self, l):
        """ Returns the block matrix at index l as sparse matrix.

        Parameters
        ----------
        l (int): Index of the matrix

        Returns
        -------
        (scipy.sparse.csr_matrix): block_matrix in a sparse data format
        """
        if l == 1:
            k = np.array([-np.ones(self.n-2), np.full((self.n-1), 2*self.d), -np.ones(self.n-2)])
            offset = [-1, 0, 1]
            A_1 = sps.diags(k, offset).toarray()
            return sps.csr_matrix(A_1)
        else:
            if self.n == 2:
                A_l = sps.csr_matrix([2*l])
            else:
                A_prev = self.get_A_l(l-1)
                dim = A_prev.shape[0]
                I_neg = -1*sps.identity(dim, format='csr')
                zeroes = sps.csr_matrix((dim, dim))

                A_l = sps.hstack([A_prev, I_neg], format='csr')

                for i in range(((self.n-1)) - 2):
                    A_l = sps.hstack([A_l, zeroes], format='csr')

                for i in range((self.n-1) - 2):
                    A_row = sps.csr_matrix((dim, 0))
                    for j in range(i):     # pylint: disable=unused-variable
                        A_row = sps.hstack([A_row, zeroes], format='csr')
                    A_row = sps.hstack([A_row, I_neg, A_prev, I_neg], format='csr')
                    for j in range(((self.n-1) - 3) - i):
                        A_row = sps.hstack([A_row, zeroes], format='csr')
                    A_l = sps.vstack([A_l, A_row], format='csr')

                A_row = sps.csr_matrix((dim, 0))

                for i in range((self.n-1) - 2):
                    A_row = sps.hstack([A_row, zeroes], format='csr')

                A_row = sps.hstack([A_row, I_neg, A_prev], format='csr')
                A_l = sps.vstack([A_l, A_row], format='csr')

            return A_l

    def get_sparse(self):
        """ Returns the block matrix as sparse matrix.

        Returns
        -------
        (scipy.sparse.csr_matrix): block_matrix in a sparse data format
        """
        return self.get_A_l(self.d)

    def eval_zeros(self):
        """ Returns the (absolute and relative) numbers of (non-)zero elements
        of the matrix. The relative number of the (non-)zero elements are with
        respect to the total number of elements of the matrix.

        Returns
        -------
        (int): number of non-zeros,
        (int): number of zeros,
        (float): relative number of non-zeros,
        (float): relative number of zeros
        """
        sparse_matrix = self.get_sparse()
        abs_values = sparse_matrix.shape[0] * sparse_matrix.shape[1]

        abs_non_zero = sparse_matrix.count_nonzero()
        abs_zero = abs_values - abs_non_zero
        rel_non_zero = abs_non_zero/abs_values
        rel_zero = abs_zero/abs_values

        return (abs_non_zero, abs_zero, rel_non_zero, rel_zero)


def main():
    """ Main function to test the BlockMatrix class.
    """
    A_d = BlockMatrix(3, 3)
    print(A_d.get_sparse().todense())
    print(A_d.get_sparse())
    print(A_d.eval_zeros())

if __name__ == "__main__":
    main()
