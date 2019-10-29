"""
Authors: Bressler_Marisa, Jeschke_Anne
Date: 2019_10_24
"""

import numpy as np
from matplotlib import use
use('qt4agg')
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2

class FiniteDifference:
    """ Represents the first and second order finite difference approximation
    of a function and allows for a computation of error to the exact
    derivatives.

    Parameters
    ----------
    h (float): Stepsize of the approximation.
    f (callable): Function to approximate the derivatives of.
    d_f (callable, optional): The analytic first derivative of `f`.
    dd_f (callable, optional): The analytic second derivative of `f`.

    Attributes
    ----------
    h (float): Stepsize of the approximation.
    """

    def __init__(self, h, f, d_f=None, dd_f=None):
        self.h = h       # pylint: disable=invalid-name
        self.f = f       # pylint: disable=invalid-name
        self.d_f = d_f   # pylint: disable=invalid-name
        self.dd_f = dd_f # pylint: disable=invalid-name

    def first_finite_diff(self, x, h=None): # pylint: disable=invalid-name
        """ Calculates the value of the first finite difference of f at x.

        Parameters
        ----------
        x (float): Point at which to calculate the first finite difference.
        h (float, optional): Stepsize of the approximation.
                             If h was not provided, use self.h.

        Returns
        -------
        (float): The value of the first finite difference of f at x.
        """
        if h is None:
            h = self.h
        return (self.f(x+h)-self.f(x))/h

    def second_finite_diff(self, x, h=None): # pylint: disable=invalid-name
        """ Calculates the value of the second finite difference of f at x.

        Parameters
        ----------
        x (float): Point at which to calculate the second finite difference.
        h (float, optional): Stepsize of the approximation.
                             If h was not provided, use self.h.

        Returns
        -------
        (float): The value of the second finite difference of f at x.
        """
        if h is None:
            h = self.h
        return (self.f(x+h)-2*self.f(x)+self.f(x-h))/(h**2)

    def compute_errors(self, a, b, p, h=None):  # pylint: disable=invalid-name
        """ Calculates an approximation to the errors between an approximation
        and the exact derivative for first and second order derivatives in the
        maximum norm.

        Parameters
        ----------
        a, b (float): Start and end point of the interval.
        p (int): Number of points plus 1 used in the approximation of the maximum norm.
        h (float, optional): Stepsize of the approximation for first and second order
                             derivatives. If h was not provided, use self.h.

        Returns
        -------
        (list of two floats):
        [maximum of errors of the approximation of the first derivative,
         maximum of errors of the approximation of the second derivative]

        Raises
        ------
        ValueError
            If no analytic derivative was provided by the user.
        """
        if h is None:
            h = self.h

        if self.d_f is None or self.dd_f is None:
            raise ValueError('One of the analytic derivatives was provided.')
        domain = np.linspace(a, b, p+1)
        err_first_diff = [abs(self.d_f(x)-self.first_finite_diff(x, h)) for x in domain]
        err_second_diff = [abs(self.dd_f(x)-self.second_finite_diff(x, h)) for x in domain]

        return (max(err_first_diff), max(err_second_diff))

    def plot_functions(self, a, b, p): # pylint: disable=invalid-name
        """ Plots f, its first and second finite difference and its first and second
        analytic derivative (if provided) on the interval [a,b].

        Parameters
        ----------
        a, b (float): Start and end point of the interval.
        p (int): Number of points plus 1 on the interval to plot.
        """
        domain = np.linspace(a, b, p+1)

        values_f = [self.f(x) for x in domain]
        values_first_diff = [self.first_finite_diff(x) for x in domain]
        values_second_diff = [self.second_finite_diff(x) for x in domain]

        plt.plot(domain, values_f, label='$f$', color='orangered')

        if self.d_f is not None:
            values_d_f = [self.d_f(x) for x in domain]
            plt.plot(domain, values_d_f, label='$f\'$', color='lightgreen')
        plt.plot(domain, values_first_diff, label='$D_h^{(1)}f$', color='green', linestyle='-.')

        if self.dd_f is not None:
            values_dd_f = [self.dd_f(x) for x in domain]
            plt.plot(domain, values_dd_f, label='$f\'\'$', color='lightskyblue')
        plt.plot(domain, values_second_diff, label='$D_h^{(2)}f$', color='blue', linestyle='-.')

        plt.legend(loc='lower right')
        plt.show()
        plt.figure()

    def plot_errors(self, a, b, p, stepsizes): # pylint: disable=invalid-name
        """ Plots maxima of errors for first and second derivative
        for a given collection of stepsizes.

        Parameters
        ----------
        a, b (float): Start and end point of the interval.
        stepsizes (list of floats): Collection of stepsizes.
        """
        errors_1 = [self.compute_errors(a, b, p, h)[0] for h in stepsizes]
        errors_2 = [self.compute_errors(a, b, p, h)[1] for h in stepsizes]
        h_2 = [h**2 for h in stepsizes]
        h_3 = [h**3 for h in stepsizes]

        plt.loglog(stepsizes, stepsizes, label='$h$',
                   color='lightgray')
        plt.loglog(stepsizes, h_2, label='$h^2$',
                   color='lightgray', linestyle='--')
        plt.loglog(stepsizes, h_3, label='$h^3$',
                   color='lightgray', linestyle=':')
        plt.loglog(stepsizes, errors_1, label='$e_f^{(1)}$', color='green')
        plt.loglog(stepsizes, errors_2, label='$e_f^{(2)}$', color='blue')

        plt.legend(loc='lower right')
        plt.show()
        plt.figure()

def main():
    """ Main function to test the FiniteDiffernce class.
    """
    g_1 = FiniteDifference(0.5, lambda x: np.sin(x)/x,
                           lambda x: (x*np.cos(x)-np.sin(x))/(x**2),
                           lambda x: -(2*x*np.cos(x)+(-2+x**2)*np.sin(x))/(x**3))
    g_1.plot_functions(np.pi, 3*np.pi, 1000)
    g_1.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])


if __name__ == "__main__":
    main()