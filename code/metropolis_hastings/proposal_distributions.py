""" Example proposal distributions.
"""
import numpy as np
from metropolis_hastings.distributions import multivariate_gaussian as log_mg


def gaussian_proposal(x_0, a=1.):
    """ Spherical gaussian proposal distribution

        Parameters
        ----------
        x_0: numpy.array
            start point
        a: float
            stepping lengthscale

        Returns
        -------
        q_rat: float
            ratio q(x_0|x_1) / q(x_1|x_0)
        x_1: numpy.array
            proposal point drawn from the chosen distribution q(x_1|x_0)
    """

    # step drawn from a unit spherical gaussian distribution
    dx = np.random.randn(len(x_0))

    # new proposed point
    x_1 = x_0 + a * dx

    # acceptance ratio modification is 1, since proposal dxn is symmetric
    q_rat = 1.

    return x_1, q_rat

def multivar_gaussian_proposal(x_0, cov):
    """ Spherical gaussian proposal distribution

        Parameters
        ----------
        x_0: numpy.array
            start point
        cov: numpy array
            covariance

        Returns
        -------
        q_rat: float
            ratio q(x_0|x_1) / q(x_1|x_0)
        x_1: numpy.array
            proposal point drawn from the chosen distribution q(x_1|x_0)
    """
    mean = np.zeros(len(x_0))

    # step drawn from a unit spherical gaussian distribution
    dx = np.random.multivariate_normal(mean, cov)

    # new proposed point
    x_1 = x_0 + dx

    # acceptance ratio modification is 1, since proposal dxn is symmetric
    q_rat = 1.

    return x_1, q_rat
