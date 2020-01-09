"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2020_01_17

Main program to demonstrate the functionality of our modules.
"""

# pylint: disable=invalid-name, superfluous-parens, import-error, unused-import

import sys
import numpy as np
import scipy as sc
import scipy.linalg as lina
# from matplotlib import use
# use('qt4agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.rcParams['font.size'] = 12

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
    R = get_qr(A)[1]

    for i in range(R.shape[1]):
        if R[i, i] == 0:
            return False

    return True

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
        raise Exception('A does not have a full column rank.')

    qr = get_qr(A)
    z = np.resize(np.dot(qr[0].transpose(), b), A.shape[1])
    r = np.resize(qr[1], (A.shape[1], qr[1].shape[1]))

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
        raise Exception('A does not have a full column rank.')

    z = np.dot(sc.transpose(get_qr(A)[0]), b).resize(A.shape[0]-A.shape[1])
    return lina.norm(z, 2)

def get_cond_A(A):
    """
    Function to get the condition of A.

    Parameters
    ----------
    A: array_like

    Returns
    -------
    float
        condition of A.
    """
    return lina.norm(A)*lina.norm(lina.inv(A))

def get_cond_ATA(A):
    """
    Function to get the condition of A transposed times A.

    Parameters
    ----------
    A: array_like

    Returns
    -------
    float
        condition of A transposed times A.
    """
    return lina.norm(np.dot(A, A.transpose()))*lina.norm(lina.inv(np.dot(A, A.transpose())))

def read_input(filename, selection=[], number_of_columns=3): # pylint: disable=dangerous-default-value
    """
    Function to read the input file containig the data to analyze.

    Parameters
    ----------
    filename: string
        Name of the file containig the data.
    selection: optional, collection
        collection of integers representig the datapoints to select,
        starting with index 0.
    number_of_columns: optional, int
        number of columns in the input file.

    Returns
    -------
    numpy.ndarray
        input data
    """
    input_file = open(filename)

    data = np.ndarray((0, number_of_columns))

    if selection:
        index = 0
        for line in input_file:
            if index in selection:
                numbers = [list(map(float, line.rstrip().split(', ')))]
                data = np.append(data, numbers, axis = 0) #pylint: disable=bad-whitespace
            index = index+1
    else:
        for line in input_file:
            numbers = [list(map(float, line.rstrip().split(', ')))]
            data = np.append(data, numbers, axis=0) #pylint: disable=bad-whitespace
    input_file.close()

    return data

def create_lgs(data, number_of_unknowns):
    """
    Creates matrix A and right-hand-side b of the linear regression system from the input data.

    Parameters
    ----------
    data: np.ndarray
        Input data as read from file.
    number_of_unknowns: int
        number of unknowns in the lgs.

    Returns
    -------
    numpy.ndarray
        matrix A
    numpy.ndarray
        vector b
    """
    A = np.ndarray((0, number_of_unknowns))
    b = np.ndarray((0))
    for line in data:
        row = np.append(line[1:number_of_unknowns], 1)
        A = np.append(A, [row], axis=0)
        b = np.append(b, line[0])

    return A, b

def plot_result(data):
    """
    Plots results of simple linear regression from the input data.

    Parameters
    ----------
    data: np.ndarray
        Input data as read from file.
    """
    A, b = create_lgs(data, 2)
    c, d = solve_qr(A, b)

    p_1 = np.linspace(75, 300, 10)
    p_0 = c*p_1+d

    plt.plot(p_1, p_0, label='$c*p_1+d$', linestyle='-', color='blue')
    plt.plot(A[:, 0], b, 'o', label='input data', color='magenta')

    plt.xlabel('$p_1$')
    plt.ylabel('$p_0$')
    plt.title('simple linear regression')
    plt.legend()
    plt.grid()
    plt.show()

def plot_result_multi(data):
    """
    Plots results of simple linear regression from the input data.

    Parameters
    ----------
    data: np.ndarray
        Input data as read from file.
    """
    A, b = create_lgs(data, 3)
    c, d, e = solve_qr(A, b)

    x = np.linspace(75, 300, 10)
    X, Y = np.meshgrid(x, x)

    Z = c*X+d*Y+e

    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, label='approximation', cmap='winter', alpha=0.5)

    ax.scatter3D(A[:, 0], A[:, 1], b, label='exact data points', color='magenta')

    ax.set_xlabel('$p_1$')
    ax.set_ylabel('$p_2$')
    ax.set_zlabel('$p_0$')
    plt.title('multilinear regression')
    ax.grid()
    plt.show()

def main():
    """ Main function to demonstrate the functionality of our modules.
    """
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        raise Exception('No input data given.')

    data = read_input(filename)
    A, b = create_lgs(data, 3)
    print(solve_qr(A, b))
    plot_result(data)
    plot_result_multi(data)



if __name__ == "__main__":
    main()
