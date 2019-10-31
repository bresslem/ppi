"""
Authors: Bressler_Marisa, Jeschke_Anne
Date: 2019_10_30
"""

import numpy as np
from matplotlib import use
#use('qt4agg')
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
    name (string): Name of function f.
    d_f (callable, optional): The analytic first derivative of `f`.
    dd_f (callable, optional): The analytic second derivative of `f`.

    Attributes
    ----------
    h (float): Stepsize of the approximation.
    """

    def __init__(self, h, f, name, d_f=None, dd_f=None):
        self.h = h       # pylint: disable=invalid-name
        self.f = f       # pylint: disable=invalid-name
        self.name = name       # pylint: disable=invalid-name
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
            At least one of the analytic derivatives was not provided by the user.
        """

        if self.d_f is None or self.dd_f is None:
            raise ValueError('At least one of the analytic derivatives was not provided.')
        if h is None:
            h = self.h
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

        plt.plot(domain, values_f, label='$g$', color='orangered')
        plt.xlabel('$x$')
        plt.ylabel('$y$')
        plt.title('Funktionenplot von ' + self.name +
                  ',\nSchrittweite $h=$' + str(self.h))

        if self.d_f is not None:
            values_d_f = [self.d_f(x) for x in domain]
            plt.plot(domain, values_d_f, label='$g\'$', color='lightgreen')
        plt.plot(domain, values_first_diff, label='$D_h^{(1)}g$', color='green', linestyle='-.')

        if self.dd_f is not None:
            values_dd_f = [self.dd_f(x) for x in domain]
            plt.plot(domain, values_dd_f, label='$g\'\'$', color='lightskyblue')
        plt.plot(domain, values_second_diff, label='$D_h^{(2)}g$', color='blue', linestyle='-.')

        plt.legend(loc='lower right')
        plt.show()
        plt.figure()

    def plot_errors(self, a, b, p, stepsizes): # pylint: disable=invalid-name
        """ Plots errors for first and second derivative for a given
        collection of stepsizes.

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
        plt.loglog(stepsizes, errors_1, label='$e_g^{(1)}$', color='green')
        plt.loglog(stepsizes, errors_2, label='$e_g^{(2)}$', color='blue')

        plt.xlabel('$h$')
        plt.ylabel('$e_g(h)$')
        plt.title('Fehlerplot von ' + self.name)
        plt.legend(loc='lower right')
        plt.show()
        plt.figure()

def main():
    """ Main function to test the FiniteDiffernce class.
    """

# Aufgabe 1.3 Teil b

    g_1 = FiniteDifference(np.pi/5, lambda x: np.sin(x)/x,'$g_1$',
                           lambda x: (x*np.cos(x)-np.sin(x))/(x**2),
                           lambda x: -(2*x*np.cos(x)+(-2+x**2)*np.sin(x))/(x**3))
#    g_1.plot_functions(np.pi, 3*np.pi, 1000)

    g_1.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])


    g_e = FiniteDifference(np.pi/5, lambda x: np.sin(2*x)/x,'$g_e$',
                           lambda x: (2*x*np.cos(2*x)-(np.sin(2*x)))/(x**2),
                           lambda x: -(4*x*np.cos(2*x)+(-2+(4*(x**2)))*np.sin(2*x))/(x**3))
#    g_e.plot_functions(np.pi, 3*np.pi, 1000)

    g_e.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    g_f = FiniteDifference(np.pi/5, lambda x: np.sin(4*x)/x,'$g_f$',
                           lambda x: (4*x*np.cos(4*x)-(np.sin(4*x)))/(x**2),
                           lambda x: -(8*x*np.cos(4*x)+(-2+(16*(x**2)))*np.sin(4*x))/(x**3))
#    g_f.plot_functions(np.pi, 3*np.pi, 1000)

    g_f.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    g_g = FiniteDifference(np.pi/5, lambda x: np.sin(10*x)/x,'$g_g$',
                           lambda x: (10*x*np.cos(10*x)-(np.sin(10*x)))/(x**2),
                           lambda x: -(20*x*np.cos(10*x)+(-2+(100*(x**2)))*np.sin(10*x))/(x**3))
#    g_g.plot_functions(np.pi, 3*np.pi, 1000)

    g_g.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    g_h = FiniteDifference(np.pi/5, lambda x: np.sin(20*x)/x,'$g_h$',
                           lambda x: (20*x*np.cos(20*x)-(np.sin(20*x)))/(x**2),
                           lambda x: -(40*x*np.cos(20*x)+(-2+(400*(x**2)))*np.sin(20*x))/(x**3))
#    g_h.plot_functions(np.pi, 3*np.pi, 1000)

    g_h.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    g_i = FiniteDifference(np.pi/5, lambda x: np.sin(100*x)/x,'$g_i$',
                           lambda x: (100*x*np.cos(100*x)-(np.sin(100*x)))/(x**2),
                           lambda x: -(200*x*np.cos(100*x)+(-2+(10000*(x**2)))*np.sin(100*x))/(x**3))
#    g_i.plot_functions(np.pi, 3*np.pi, 1000)

    g_i.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    g_j = FiniteDifference(np.pi/5, lambda x: np.sin(200*x)/x,'$g_j$',
                           lambda x: (200*x*np.cos(200*x)-(np.sin(200*x)))/(x**2),
                           lambda x: -(400*x*np.cos(200*x)+(-2+(40000*(x**2)))*np.sin(200*x))/(x**3))
#    g_j.plot_functions(np.pi, 3*np.pi, 1000)

    g_j.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    g_k = FiniteDifference(np.pi/5, lambda x: np.sin(300*x)/x,'$g_k$',
                           lambda x: (300*x*np.cos(300*x)-(np.sin(300*x)))/(x**2),
                           lambda x: -(600*x*np.cos(300*x)+(-2+(90000*(x**2)))*np.sin(300*x))/(x**3))
#    g_k.plot_functions(np.pi, 3*np.pi, 1000)

    g_k.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    g_l = FiniteDifference(np.pi/5, lambda x: np.sin(400*x)/x,'$g_l$',
                           lambda x: (400*x*np.cos(400*x)-(np.sin(400*x)))/(x**2),
                           lambda x: -(800*x*np.cos(400*x)+(-2+(160000*(x**2)))*np.sin(400*x))/(x**3))
#    g_l.plot_functions(np.pi, 3*np.pi, 1000)

    g_l.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    g_m = FiniteDifference(np.pi/5, lambda x: np.sin(450*x)/x,'$g_m$',
                           lambda x: (450*x*np.cos(450*x)-(np.sin(450*x)))/(x**2),
                           lambda x: -(900*x*np.cos(450*x)+(-2+(202500*(x**2)))*np.sin(450*x))/(x**3))
#    g_m.plot_functions(np.pi, 3*np.pi, 1000)

    g_m.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    g_n = FiniteDifference(np.pi/5, lambda x: np.sin(475*x)/x,'$g_n$',
                           lambda x: (475*x*x*np.cos(475*x)-(np.sin(475*x)))/(x**2),
                           lambda x: -(950*x*np.cos(475*x)+(-2+(225625*(x**2)))*np.sin(475*x))/(x**3))
#    g_n.plot_functions(np.pi, 3*np.pi, 1000)

    g_n.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])

    g_o = FiniteDifference(np.pi/5, lambda x: np.sin(500*x)/x,'$g_o$',
                           lambda x: (500*x*np.cos(500*x)-(np.sin(500*x)))/(x**2),
                           lambda x: -(1000*x*np.cos(500*x)+(-2+(250000*(x**2)))*np.sin(500*x))/(x**3))
#    g_o.plot_functions(np.pi, 3*np.pi, 1000)

    g_o.plot_errors(np.pi, 3*np.pi, 1000, [1e-8, 1e-7, 1e-6, 1e-5,
                                           1e-4, 1e-3, 1e-2, 1e-1, 1, 10])


if __name__ == "__main__":
    main()
