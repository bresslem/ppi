"""
Author: Jeschke_Anne
Date: 2019_10_20
"""

# Damit Plots auf den Mathe-Rechnern angezeigt werden koennen
from matplotlib import use
# use("qt4Agg")
# Importiere Bibliothek zum Plotten
import matplotlib.pyplot as plt
# Aendere ein paar Einstellungen fuer bessere Darstellung der Plots
plt.rcParams['font.size'] = 12
plt.rcParams['lines.linewidth'] = 2
# Importiert Bibliothek fuer numerische Methoden, Klassen etc.
import numpy as np
#===============================================================================
# Ende Kopfzeile
#===============================================================================

class FiniteDifference:
    """ Represents the first and second order finite difference approximation
    of a function and allows for a computation of error to the exact
    derivatives.

    Parameters
    ----------
    h : float
        Stepsize of the approximation.
    f : callable
        Function to approximate the derivatives of.
    d_f : callable, optional
        The analytic first derivative of `f`.
    dd_f : callable, optional
        The analytic second derivative of `f`.

    Attributes
    ----------
    h : float
        Stepsize of the approximation.
    """

    def __init__(self, h, f, d_f=None, dd_f=None):
        self.h = h
        self.f = f
        self.d_f = d_f
        self.dd_f = dd_f

    def firstFD(self, x):
        return (self.f(x+self.h)-self.f(x))/self.h

    def secondFD(self, x):
        return (self.f(x+self.h) - 2*self.f(x) + self.f(x-self.h)) / (self.h**2)

    def compute_errors(self, a, b, p):  # pylint: disable=invalid-name
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

        import numpy as np

        if self.d_f!=None:
            a1 = np.zeros(p+1)
            for i in range(p+1):
                xi = a + i*abs(b-a)/p
                a1[i] = self.d_f(xi) - FiniteDifference.firstFD(self, xi)
            e1 = np.max(a1)

        else:
            print ("The first analytic derivative of $f$ was not provided!")

        if self.dd_f!=None:
            a2 = np.zeros(p+1)
            for i in range(p+1):
                xi = a + i*abs(b-a)/p
                a2[i] = self.d_f(xi) - FiniteDifference.secondFD(self, xi)
            e2 = np.max(a2)

        else:
            print ("The second analytic derivative of $f$ was not provided!")

        if self.d_f!=None and self.dd_f!=None:
            return np.array([e1, e2])

    def draw_functions(self, a, b, p):
        import numpy as np
        import matplotlib.pyplot as plt

        x = np.zeros(p+1)
        for i in range(p+1):
            x[i] = a + i*abs(b-a)/p
        y_f = self.f(x)
        y_firstFD = FiniteDifference.firstFD(self, x)
        y_secondFD = FiniteDifference.secondFD(self, x)

        plt.plot(x, y_f, label="$f$")
        plt.plot(x, y_firstFD, label="first finite difference of $f$")
        plt.plot(x, y_secondFD, label="second finite difference of $f$")

        if self.d_f!=None:
            y_d_f = self.d_f(x)
            plt.plot(x, y_d_f, label="first derivative of $f$")
        else:
            print ("The first analytic derivative of $f$ was not provided!")

        if self.dd_f!=None:
            y_dd_f = self.dd_f(x)
            plt.plot(x, y_dd_f, label="second derivative of $f$")
        else:
            print ("The second analytic derivative of $f$ was not provided!")

        plt.legend()
        plt.show()

    def draw_errors(self, a, b, p, h_array):
        import numpy as np
        import matplotlib.pyplot as plt

        if self.d_f!=None:
            e1_h = np.zeros(len(h_array))
            for j in range(len(h_array)):
                a1 = np.zeros(p+1)
                h = h_array[j]
                for i in range(p+1):
                    xi = a + i*abs(b-a)/p
                    a1[i] = self.d_f(xi) - ((self.f(xi+h)-self.f(xi))/h)
                e1_h[j] = np.max(a1)
            plt.loglog(h_array, e1_h, label="errors of first finite difference of $f$")

        else:
            print ("The first analytic derivative of $f$ was not provided!")

        if self.dd_f!=None:
            e2_h = np.zeros(len(h_array))
            for j in range(len(h_array)):
                a2 = np.zeros(p+1)
                h = h_array[j]
                for i in range(p+1):
                    xi = a + i*abs(b-a)/p
                    a2[i] = self.dd_f(xi) - ((self.f(xi+h) - 2*self.f(xi) + self.f(xi-h)) / (h**2))
                e2_h[j] = np.max(a2)
            plt.loglog(h_array, e2_h, label="errors of second finite difference of $f$")

        else:
            print ("The first analytic derivative of $f$ was not provided!")

        plt.loglog(h_array, h_array, color="lightgray", label="first order", linestyle="-")

        secondorder_array = h_array**2
        plt.loglog(h_array, secondorder_array, color="lightgray", label="second order", linestyle="--")

        thirdorder_array = h_array**3
        plt.loglog(h_array, thirdorder_array, color="lightgray", label="third order", linestyle=":")

        plt.legend()
        plt.show()


def main():
    """ Presents a quick example ... (TODO) ...

    """
    import numpy as np

    def g1(x):
        return np.sin(x)/x

    def d_g1(x):
        return np.cos(x)/x - np.sin(x)/(x**2)

    def dd_g1(x):
        return -np.sin(x)/x - 2*np.cos(x)/(x**2) + 2*np.sin(x)/(x**3)

    FD = FiniteDifference(0.1, g1, d_g1, dd_g1)

    print(FD.compute_errors(np.pi, 3*np.pi, 1000))

    FD.draw_functions(np.pi, 3*np.pi, 1000)

    FD.draw_errors(np.pi, 3*np.pi, 1000, np.array([1e-4, 1e-3, 1e-2, 1e-1, 1, 10]))


if __name__ == "__main__":
    main()
