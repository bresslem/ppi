"""
Authors: Bressler_Marisa, Jeschke_Anne
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
    h_string (string): Stepsize h as string.
    f (callable): Function to approximate the derivatives of.
    name (string): Name of function f.
    d_f (callable, optional): The analytic first derivative of `f`.
    dd_f (callable, optional): The analytic second derivative of `f`.

    Attributes
    ----------
    h (float): Stepsize of the approximation.
    """

    def __init__(self, h, h_string, f, name_f, d_f=None, dd_f=None):
        self.h = h                  # pylint: disable=invalid-name
        self.h_string = h_string    # pylint: disable=invalid-name
        self.f = f                  # pylint: disable=invalid-name
        self.name_f = name_f        # pylint: disable=invalid-name
        self.d_f = d_f              # pylint: disable=invalid-name
        self.dd_f = dd_f            # pylint: disable=invalid-name

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
        plt.title('Funktionenplot von ' + str(self.name_f) +
                  ', Schrittweite $h=$' + self.h_string)

        if self.d_f is not None:
            values_d_f = [self.d_f(x) for x in domain]
            plt.plot(domain, values_d_f, label='$g\'$', color='lightgreen')
        plt.plot(domain, values_first_diff, label='$D_h^{(1)}g$', color='green', linestyle='-.')

        if self.dd_f is not None:
            values_dd_f = [self.dd_f(x) for x in domain]
            plt.plot(domain, values_dd_f, label='$g\'\'$', color='lightskyblue')
        plt.plot(domain, values_second_diff, label='$D_h^{(2)}g$', color='blue', linestyle='-.')

        plt.legend(loc='lower right')
        plt.grid()
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
        plt.title('Fehlerplot von ' + str(self.name_f))
        plt.legend(loc='lower right')
        plt.grid()
        plt.show()
        plt.figure()

def main():
    """ Main function to test the FiniteDiffernce class.
    """

# Aufgabe 1.3

# a) Fehlerplot für in x-Richtung gestreckte und in y-Richtung gestauchte Funktion
#    zur Veranschaulichung der Entwicklung des Graphen auch Funktionenplot

    js = [1, 0.75, 0.5, 0.25, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]

    for j in js:
        g_j = FiniteDifference(np.pi/10, '$\pi/10$', lambda x: np.sin(j*x)/x,
                               '$g_j$ mit $j=$' + str(j),
                               lambda x: (j*x*np.cos(j*x)-np.sin(j*x))/(x**2),
                               lambda x: (-2*j*x*np.cos(j*x)+(2-j**2*x**2)*np.sin(j*x))/(x**3))

        #g_j.plot_functions(np.pi, 3*np.pi, 1000)

        g_j.plot_errors(np.pi, 3*np.pi, 1000,
                        [1e-10, 5e-10, 1e-9, 5e-9, 1e-8, 5e-8, 1e-7, 5e-7,
                         1e-6, 5e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3,
                         1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10])


if __name__ == "__main__":
    main()
