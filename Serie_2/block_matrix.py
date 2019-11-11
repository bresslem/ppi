"""

Author: Bressler_Marisa, Jeschke_Anne
Date:
"""

from scipy.sparse import csr_matrix
from numpy import array
from numpy import count_nonzero

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
        self.d = d      # pylint: disable=invalid-name
        self.n = n      # pylint: disable=invalid-name

    def get_sparse(self):
        """ Returns the block matrix as sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            block_matrix in a sparse data format
        """
        matrices = []

        data = np.full((3*self.n-5), -1)
        row = np.zeros(3*self.n-5)
        row[-1] = self.n-2
        row[-2] = self.n-2
        col = np.zeros(3*self.n-5)
        col[-1] = self.n-2
        col[-2] = self.n-1

        for i in range(3*self.n-4):
            if i%3 == 0:
                data[i] = 2*self.d
            if i%3 == 0 and i!=(3*self.n-5):
                row[i-1] = i/3
                row[i] = i/3
                row[i+1] = i/3
                col[i-1] = i/3 - 1
                col[i] = i/3
                col[i+1] = i/3 + 1

        #2d, -1, -1, 2d, -1, -1, 2d, -1, ...,  -1,  2d
        # 0,  0,  1,  1,  1,  2,  2,  2, ..., n-2, n-2
        # 0,  1,  0,  1,  2,  1,  2,  3, ..., n-1, n-2

        A_1 = csr_matrix((data, (row, col)), shape=(self.n-1,self.n-1))

        matrices.append(A_1)




        return matrices[-1]

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