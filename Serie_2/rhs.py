"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2019_11_26
"""
# pylint: disable=invalid-name
import scipy.sparse as sps
import numpy as np
import block_matrix

def rhs(d, n, f):
    """ Computes the right-hand side vector `b` for a given function `f`.
    Parameters
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is `f(x)`.
        Here `x` is a scalar or array_like of `numpy`. The return value is a scalar.
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
        for i in range(1, (n-1)+1):
            rhs_vector[i-1] = f(i/n)
    else:
        for i in range(1, ((n-1)**d)+1):
            x = np.zeros(d)
            s = i
            for l in range (2, d+1):
#                print("s is", s)
                if s%((n-1)**(d-l+1)) == 0:
                    x[l-1] = (s//((n-1)**(d-l+1)))/n
                    s = (n-1)**(d-l+1)
                else:
                    x[l-1] = (s//((n-1)**(d-l+1))+1)/n
                    s = s%((n-1)**(d-l+1))
#                print("x[", l-1, "]=", x[l-1], "and the new i is", i)
#            print("i is", i)
            x[0] = s/n
            print(x*4)
            rhs_vector[i-1] = f(x)
    print(rhs_vector)
    return rhs_vector

def f(x):
    return(4*x[0]+3*(4*x[1]-1)+9*(4*x[2]-1))
#    return(4*x[0]+3*(4*x[1]-1))

def main():
    """ Main function to test rhs
    """
    print(rhs(3, 4, f))

if __name__ == "__main__":
    main()