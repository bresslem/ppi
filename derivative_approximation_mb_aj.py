"""
Authors: Bressler_Marisa, Jeschke_Anne
Date: 2019_10_22
"""

# from matplotlib import use
# use("qt4Agg")
import numpy as np
from matplotlib.pyplot import plot, show, legend, loglog, figure


class FiniteDifference:
    """ Represents the first and second order finite difference approximation
    of a function and allows for a computation of error to the exact
    derivatives.

    Parameters
    ----------
    h : float
        Stepzise of the approximation.
    f : callable
        Function to approximate the derivatives of.
    d_f : callable, optional
        The analytic first derivative of `f`.
    dd_f : callable, optional
        The analytic second derivative of `f`.

    Attributes
    ----------
    h : float
        Stepzise of the approximation.
    """

    def __init__(self, h, f, d_f=None, dd_f=None):
        self.h = h       # pylint: disable=invalid-name
        self.f = f       # pylint: disable=invalid-name
        self.d_f = d_f   # pylint: disable=invalid-name
        self.dd_f = dd_f # pylint: disable=invalid-name

    def first_finite_diff(self, x, h=None): # pylint: disable=invalid-name
        """ Calculates the value of the first finite difference of f.

        Parameters
        ----------
        x : float
            Point at which to calculate the first finite difference.

        Returns
        -------
        float
            The value of the first finite difference of f at x.
        """
        if h is None:
            h = self.h
        return (self.f(x+h)-self.f(x))/h

    def second_finite_diff(self, x, h=None): # pylint: disable=invalid-name
        """ Calculates the value of the second finite difference of f.

        Parameters
        ----------
        x : float
            Point at which to calculate the second finite difference.

        Returns
        -------
        float
            The value of the second finite difference of f at x.
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
        a, b : float
            Start and end point of the interval.
        p : int
            Number of points used in the approximation of the maximum norm.

        Returns
        -------
        float
            Errors of the approximation of the first derivative.
        float
            Errors of the approximation of the second derivative.

        Raises
        ------
        ValueError
            If no analytic derivative was provided by the user.
        """
        if self.d_f is None or self.dd_f is None:
            raise Exception('ValueError: No analytic derivative was provided.')
        if h is None:
            h = self.h
        domain = np.linspace(a, b, p)
        first_diff = [abs(self.d_f(x)-self.first_finite_diff(x, h)) for x in domain]
        second_diff = [abs(self.dd_f(x)-self.second_finite_diff(x, h)) for x in domain]

        return (max(first_diff), max(second_diff))

    def plot_functions(self, a, b, p): # pylint: disable=invalid-name
        """ Plots f, its first and second finite difference and first and second
        analytic derivative on the interval [a,b].

        Parameters
        ----------
        a, b : float
            Start and end point of the interval.
        p : int
            Number of points on the interval to plot.
        """
        domain = np.linspace(a, b, p)

        values_f = [self.f(x) for x in domain]
        values_first_diff = [self.first_finite_diff(x) for x in domain]
        values_second_diff = [self.second_finite_diff(x) for x in domain]

        plot(domain, values_f, label='$f$')
        plot(domain, values_first_diff, label='$D_h^{(1)}f$')
        plot(domain, values_second_diff, label='$D_h^{(2)}f$')

        if self.d_f is not None:
            values_d_f = [self.d_f(x) for x in domain]
            plot(domain, values_d_f, label='$f\'$')
        if self.dd_f is not None:
            values_dd_f = [self.dd_f(x) for x in domain]
            plot(domain, values_dd_f, label='$f\'\'$')
        legend(loc='lower right')
        show()

    def plot_errors(self, a, b, p, stepsizes): # pylint: disable=invalid-name
        """ Plots errors for first and second derivative for a given
        collection of stepsizes.

        Parameters
        ----------
        a, b : float
            Start and end point of the interval.
        stepsizes: list
            Collection of stepsizes.
        """
        errors_1 = [self.compute_errors(a, b, p, h)[0] for h in stepsizes]
        errors_2 = [self.compute_errors(a, b, p, h)[1] for h in stepsizes]
        h_2 = [h**2 for h in stepsizes]
        h_3 = [h**3 for h in stepsizes]
        loglog(stepsizes, errors_1, label='$e_f^{(1)}$')
        loglog(stepsizes, errors_2, label='$e_f^{(2)}$')
        loglog(stepsizes, stepsizes, label='$h$',
               color="lightgray", linestyle="-")
        loglog(stepsizes, h_2, label='$h^2$',
               color="lightgray", linestyle="--")
        loglog(stepsizes, h_3, label='$h^3$',
               color="lightgray", linestyle=":")

        legend(loc='lower right')
        show()

def main():
    """ Presents a quick example.
    """
    g_1 = FiniteDifference(np.pi/3, lambda x: np.sin(x)/x, lambda x: (x*np.cos(x)-np.sin(x))/(x**2), lambda x: -(2*x*np.cos(x)+(-2+x**2)*np.sin(x))/x**3)
    g_1.plot_functions(np.pi, 3*np.pi, 1000)
    figure()
    g_1.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])



if __name__ == "__main__":
    main()
