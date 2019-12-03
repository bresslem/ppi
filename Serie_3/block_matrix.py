"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2019_11_13

Creates the block matrix required to solve the discrete Poisson-Problem using
finite differences and analyzes the amount of space needed.
"""
import scipy.sparse as sps
import scipy.sparse.linalg as lina
import numpy as np
from matplotlib import use
#use('qt4agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2

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

    def __init__(self, d, n): #pylint: disable=invalid-name
        self.d = d #pylint: disable=invalid-name
        self.n = n #pylint: disable=invalid-name

    def get_matrix_l(self, l): #pylint: disable=invalid-name
        """ Returns the block matrix at index l as sparse matrix.

        Parameters
        ----------
        l (int): Index of the matrix

        Returns
        -------
        (scipy.sparse.csr_matrix): block_matrix in a sparse data format
        """
        if l == 1:
            # base case: creating the matrix for d=1
            k = np.array([-np.ones(self.n-2), np.full((self.n-1), 2*self.d), -np.ones(self.n-2)])
            offset = [-1, 0, 1]
            return sps.csr_matrix(sps.diags(k, offset).toarray())
        else:
            if self.n == 2:
                # if n is 2 the matrix contains only one value
                matrix_l = sps.csr_matrix([2*l])
            else:
                # recursion step: creating the previous matrix
                matrix_prev = self.get_matrix_l(l-1)
                dim = matrix_prev.shape[0]
                # creating the negativ identity matrix and a zero matrix the size
                # of the previous matrix
                identity_neg = -1*sps.identity(dim, format='csr')
                zeroes = sps.csr_matrix((dim, dim))

                # creating the first row of the block matrix, consisting of the previous matrix,
                # a negative identity and the rest are zeroes
                matrix_l = sps.hstack([matrix_prev, identity_neg], format='csr')

                for _ in range(((self.n-1)) - 2):
                    matrix_l = sps.hstack([matrix_l, zeroes], format='csr')

                # creating the following rows from the correct amount of zeroes,
                # the negative identity, the previous matrix, the negative identity again
                # and the rest zeroes and adding it to the bottom of the new matrix
                for i in range((self.n-1) - 2):
                    matrix_row = sps.csr_matrix((dim, 0))
                    for _ in range(i):
                        matrix_row = sps.hstack([matrix_row, zeroes], format='csr')
                    matrix_row = sps.hstack([matrix_row, identity_neg, matrix_prev, identity_neg],
                                            format='csr')
                    for _ in range(((self.n-1) - 3) - i):
                        matrix_row = sps.hstack([matrix_row, zeroes], format='csr')
                    matrix_l = sps.vstack([matrix_l, matrix_row], format='csr')

                # creating the last row consiting of zeroes, a negative identity and
                # the previous matrix
                matrix_row = sps.csr_matrix((dim, 0))

                for i in range((self.n-1) - 2):
                    matrix_row = sps.hstack([matrix_row, zeroes], format='csr')

                matrix_row = sps.hstack([matrix_row, identity_neg, matrix_prev], format='csr')
                matrix_l = sps.vstack([matrix_l, matrix_row], format='csr')

            return matrix_l

    def get_sparse(self):
        """ Returns the block matrix as sparse matrix.

        Returns
        -------
        (scipy.sparse.csr_matrix): block_matrix in a sparse data format
        """
        return self.get_matrix_l(self.d)

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

    def get_lu(self):
        """ Provides an LU-Decomposition of the represented matrix A of the
        form pr * A * pc = l * u

        Returns
        -------
        pr : scipy.sparse.csr_matrix
            row permutation matrix of LU-decomposition
        l : scipy.sparse.csr_matrix
            lower triangular unit diagonal matrix of LU-decomposition
        u : scipy.sparse.csr_matrix
            upper triangular matrix of LU-decomposition
        pc : scipy.sparse.csr_matrix
            column permutation matrix of LU-decomposition
        """

    def eval_zeros_lu(self):
        """ Returns the absolute and relative numbers of (non-)zero elements of
        the LU-Decomposition. The relative quantities are with respect to the
        total number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        int
            Number of zeros
        float
            Relative number of non-zeros
        float
            Relative number of zeros
        """

    def get_cond(self):
        """ Computes the condition number of the represented matrix.

        Returns
        -------
        float
            condition number with respect to max-norm
        """
        sparse_matrix = self.get_sparse().tocsc()
        return lina.norm(sparse_matrix, np.inf)*lina.norm(lina.inv(sparse_matrix), np.inf)


def plot_non_zeros(n_array):
    """
    Plots the amount of non zeros values contained in the block matrix and its LU-decomposition
    for a given array of n's.

    Parameters
    ----------
    n_array (list of ints): The n's for which to plot the non-zeroes and total values.
    """
    # pylint: disable=invalid-name
    # for d in [1, 2, 3]:
    #     non_zeros = []
    #     absolute_values = []
    #     for n in n_array:
    #         matrix = BlockMatrix(d, n)
    #         non_zeros.append(matrix.eval_zeros()[0])
    #         sparse_matrix = matrix.get_sparse()
    #         absolute_values.append(sparse_matrix.shape[0] * sparse_matrix.shape[1])
    #     plt.plot(n_array, non_zeros, "b.", label='absolute number of non zero values')
    #     plt.plot(n_array, absolute_values, "g.", label='absolute number of values')
    #     plt.xlabel('$n$')
    #     plt.title('d = ' + str(d))
    #     plt.legend()
    #     plt.grid()
    #     plt.show()


def plot_cond(n_array):
    """
    Plots the condition of the block matrix for a given array of n's.

    Parameters
    ----------
    n_array (list of ints): The n's for which to plot the non-zeroes and total values.
    """


def main():
    """ Main function to use the BlockMatrix class.
    """
    print(BlockMatrix(2, 3).get_sparse().todense())
    print(BlockMatrix(2, 3).get_cond())

if __name__ == "__main__":
    main()
