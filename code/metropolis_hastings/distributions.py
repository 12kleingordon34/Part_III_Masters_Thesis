""" Example probability distributions.
"""
import numpy as np
from numpy.linalg import inv, det
from scipy.stats import multivariate_normal

def gaussian(x, mu=0, sigma=1):
    """ Spherical multidimensional gaussian probability distribution

        Parameters
        ----------

        x:  numpy.array
            independent variable
        mu: numpy.array or float
            mean(s)
        sigma: numpy.array or float
            standard deviation(s)

        Output
        ------
        log likelihood of gaussian at position x
    """
    t = (x - mu) / sigma
    return -1.0 * np.log(2*np.pi)-np.dot(t, t)

def multivariate_gaussian(x,mu,Sigma):
    """ General multidimensional gaussian probability distribution

        Parameters
        ----------

        x:  numpy.array
            independent variable
        mu: numpy.array
            means
        Sigma: numpy.array
            Covariance matrix
            
        Output
        ------
        log likelihood of gaussian at position x

            
    """
    invSigma = inv(Sigma)
    detTwoPiSigma = det(2 * np.pi * Sigma)

    return -0.5 * np.log(detTwoPiSigma) -0.5 * (x-mu).dot(invSigma.dot(x-mu))

def bimodal_multi_gaussian(x,mu1,Sigma1,mu2,Sigma2):
    """ Bimodal multidimensional gaussian probability distribution

        Parameters
        ----------

        x:  numpy.array
            independent variable
        mu: numpy.array
            means
        Sigma: numpy.array
            Covariance matrix
            
        Output
        ------
        log likelihood of gaussian at position x

            
    """
    x = 0.5 * (multivariate_normal.pdf(x, mu1, Sigma1) + multivariate_normal.pdf(x, mu2, Sigma2))

    return np.log(x)

def Rosenbrock(x):
    """ Rosenbrock function Probability Density

        Parameters
        ----------

        x:  numpy.array
            independent variable
            
        Output
        ------
        log likelihood of gaussian at position x

            
    """
    x1 = x[0]
    x2 = x[1]

    return -20.* (x2 - x1**2) **2 + (1 - x1) **2


def toroidal_multi_gaussian(x):
    """ Bimodal multidimensional gaussian probability distribution

        Parameters
        ----------

        x:  numpy.array
            independent variable
        mu: numpy.array
            means
        Sigma: numpy.array
            Covariance matrix
            
        Output
        ------
        log likelihood of gaussian at position x

            
    """
    Mu = np.array([2*np.pi, 0.])
    Mu1 = np.array([0, 0])
    Mu2 = np.array([0, 2*np.pi])
    Mu3 = np.array([2*np.pi, 2*np.pi])
    Sigma = np.array([[0.5, 0],[0, 0.5]])
    x = 0.25 * (
        multivariate_normal.pdf(x, Mu, Sigma) 
        + multivariate_normal.pdf(x, Mu1, Sigma) 
        + multivariate_normal.pdf(x, Mu2, Sigma) 
        + multivariate_normal.pdf(x, Mu3, Sigma)
    )

    return np.log(x)
