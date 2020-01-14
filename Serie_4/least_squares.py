"""
Author: Bressler_Marisa, Jeschke_Anne
Date: 2020_01_17

Main program to demonstrate the functionality of our modules.
"""

# pylint: disable=invalid-name, import-error, unused-import, no-member

import sys
import numpy as np
import scipy as sc
import scipy.linalg as lina
import matplotlib as mpl
# mpl.use('qt4agg')
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
plt.rcParams['font.size'] = 12

def read_input(filename, selection=None, number_of_columns=3):
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
                data = np.append(data, numbers, axis=0)
            index = index+1
    else:
        for line in input_file:
            numbers = [list(map(float, line.rstrip().split(', ')))]
            data = np.append(data, numbers, axis=0)
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

def create_lgs_p2(data):
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
    A = np.ndarray((0, 2))
    b = np.ndarray((0))
    for line in data:
        row = np.append(line[2], 1)
        A = np.append(A, [row], axis=0)
        b = np.append(b, line[0])

    return A, b

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

    z = np.dot(sc.transpose(get_qr(A)[0]), b)[A.shape[1]:]

    if z.size == 0:
        return 0

    return lina.norm(z, 2)

def get_cond(A):
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
    return lina.norm(A, 2)*lina.norm(lina.pinv(A), 2)

def get_cond_transposed(A):
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
    ATA = np.dot(A, A.transpose())

    return lina.norm(ATA, 2)*lina.norm(lina.pinv(ATA), 2)

def plot_result(data_list, labels):
    """
    Plots results of simple linear regression from the input data.

    Parameters
    ----------
    data_list: np.ndarray
        list of datapoints to analyze, each with different modifications.
    labels:
        list of descriptions of the data.
    """
    mpl.style.use('classic')
    i = 0
    for data in data_list:
        A, b = create_lgs(data, 2)
        c, d = solve_qr(A, b)

        p_1 = np.linspace(75, 300, 10)
        p_0 = c*p_1+d

        cond_A = get_cond(A)
        cond_ATA = get_cond_transposed(A)
        residuum = norm_of_residuum(A, b)

        plt.plot(A[:, 0], b, '.', label='Modification %d: %s' %(i, labels[i]), color='C'+str(i))
        plt.plot(p_1, p_0, label='Linear regression of mod. %d' %i,
                 linestyle='--', color='C'+str(i))
        print('Modification %d: cond_2(A)=%f, cond_2(A^T A)=%f, ||Ax-b||_2=%f'
              %(i, cond_A, cond_ATA, residuum))
        print('Linear regression: p_0 = %f*p_1+%f' %(c, d))
        i = i+1

    plt.xlabel('$p_1$')
    plt.ylabel('$p_0$')
    plt.title('simple linear regression')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_result_p2(data):
    """
    Plots results of simple linear regression from the input data.

    Parameters
    ----------
    data_list: np.ndarray
        list of datapoints to analyze, each with different modifications.
    labels:
        list of descriptions of the data.
    """
    mpl.style.use('classic')

    A, b = create_lgs(data, 2)
    c, d = solve_qr(A, b)

    p_1 = np.linspace(50, 350, 10)
    p_0 = c*p_1+d

    cond_A = get_cond(A)
    cond_ATA = get_cond_transposed(A)
    residuum = norm_of_residuum(A, b)

    plt.plot(A[:, 0], b, '.', label='Data points p_1')
    plt.plot(p_1, p_0, label='Linear regression p_1',
             linestyle='--')
    print('p_1: cond_2(A)=%f, cond_2(A^T A)=%f, ||Ax-b||_2=%f'
          %(cond_A, cond_ATA, residuum))
    print('Linear regression: p_0 = %f*p_1+%f' %(c, d))

    A, b = create_lgs_p2(data)
    c, d = solve_qr(A, b)

    p_0 = c*p_1+d

    cond_A = get_cond(A)
    cond_ATA = get_cond_transposed(A)
    residuum = norm_of_residuum(A, b)

    plt.plot(A[:, 0], b, '.', label='Data points p_12')
    plt.plot(p_1, p_0, label='Linear regression p_2',
             linestyle='--')
    print('p_2: cond_2(A)=%f, cond_2(A^T A)=%f, ||Ax-b||_2=%f'
          %(cond_A, cond_ATA, residuum))
    print('Linear regression: p_0 = %f*p_2+%f' %(c, d))


    plt.xlabel('$p_1 und 2$')
    plt.ylabel('$p_0$')
    plt.title('simple linear regression using p_1 und p_2')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

def plot_result_multilinear(data):
    """
    Plots results of simple linear regression from the input data.

    Parameters
    ----------
    data: np.ndarray
        Input data as read from file.
    """
    mpl.style.use('default')

    A, b = create_lgs(data, 3)
    c, d, e = solve_qr(A, b)

    x = np.linspace(75, 300, 10)
    X, Y = np.meshgrid(x, x)

    Z = c*X+d*Y+e

    cond_A = get_cond(A)
    cond_ATA = get_cond_transposed(A)
    residuum = norm_of_residuum(A, b)

    print('Multilinear regression: p_0 = %f*p_1+%f*p_2+%f' %(c, d, e))
    print('cond_2(A)=%f, cond_2(A^T A)=%f, ||Ax-b||_2=%f'
          %(cond_A, cond_ATA, residuum))


    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, label='approximation', cmap='winter', alpha=0.5)

    ax.scatter3D(A[:, 0], A[:, 1], b, label='exact data points', color='red')

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
    data_list = []
    labels = []
    data_list.append(data)
    labels.append("all samples")
    data_list.append(np.append(read_input(filename), [[480, 230, 0]], axis=0))
    labels.append("one big error")
    data_list.append(read_input(filename, [0, 1, 2, 3, 4, 5]))
    labels.append("only first six")
    plot_result(data_list, labels)
    plot_result_p2(data)

    plot_result_multilinear(data)


if __name__ == "__main__":
    main()
