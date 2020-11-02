''' Functions used for calculating steps in curvelinear coordinates.
'''

import numpy as np
import acor
from scipy.stats import threshold

def parallel_trans(t, P):
    '''
    Finds the theta and phi components of the parallel transported
    vector t, transported across the great circle (theta=pi/2), from
    phi=0 to phi=phi_final

    Equations used were taken from James Wheeler's lecture course,
    Univerity of Utah State.

    http://www.physics.usu.edu/Wheeler/GenRel2013/Notes/Geodesics.pdf

    Parameters
    ----------
    t: numpy array
        vector to be parallel transported (t[0]=phi cmpt,
        t[1]=theta cmpt)
    P: numpy array
        Angular coordinates of transported vector destination

    Returns
    -------
    t_pt: numpy array
        Vector containing phi and theta components of transported 
        vector
    '''
    # Angular basis vector components at phi=0, theta=pi/2
    t_theta = t.dot(np.array([0,0,-1]))
    t_phi = t.dot(np.array([0,1,0]))   
    
    # Angular coordinates of P
    P_phi = P[0]
    P_theta = P[1]

    epsilon = np.sin(P_theta)
    zeta = np.cos(P_theta)

    # Finding transported components following Wheeler's Equations
    t_phi_pt = t_phi * np.cos(P_phi*zeta) \
               - t_theta * np.sin(P_phi*zeta) / epsilon

    t_theta_pt = t_theta * np.cos(P_phi*zeta) \
                 - t_phi * np.sin(P_phi*zeta) * epsilon
            
    t_pt = [t_phi_pt, t_theta_pt]

    return t_pt

def sph_basis(x):
    '''
    Creates a spherical vector basis set for a point
    on a sphere with angular theta, psi coordinates given
    by x

    Parameters
    ----------
    x: numpy array
        asdf

    Returns
    -------
    basis: numpy array
        Spherical vector basis set, [r, e_theta, e_psi]

    '''

    psi = x[0]
    theta = x[1]

    S_the = np.sin(theta)
    C_the = np.cos(theta)

    S_psi = np.sin(psi)
    C_psi = np.cos(psi)

    X = S_the * C_psi
    Y = S_the * S_psi
    Z = C_the

    r = [X, Y, Z]
    e_the = [C_the * C_psi, C_the * S_psi, -S_the]
    e_psi = [-S_psi, C_psi, 0.]

    basis = np.array([r, e_the, e_psi])
    basis[abs(basis) < 1e-16] = 0
    
    return np.array(basis)

def car2sph(xyz):
    '''
    Convert cartesian coordinates to spherical angles, theta and psi.

    Parameters
    ---------- 
    x: numpy array
        Cartesian coordinates

    Returns
    -------
    sph_coord: numpy array
        asdf

    '''
    sph_coord = np.empty(2)
    xy = xyz[0] **2 + xyz[1] ** 2
    sph_coord[0] = np.arctan2(xyz[1], xyz[0]) % (2 * np.pi) # y/x = tan(phi)
    sph_coord[1] = np.arctan2(np.sqrt(xy), xyz[2]) % np.pi # r/z = tan(theta)

    return sph_coord

def sph_tangent(R, e_X):
    '''
    Calculates the components of vector R in the angular spherical basis
    vectors e_theta, e_psi at point X on the spherical surface.

    Parameters
    ----------
    R: numpy array
        Vector in cartesian space
    e_X: numpy array
        Spherical basis vectors at point X

    Returns
    -------
    t: numpy array
        Normalised tangent vector to sphere at point X
    '''
    e_theta = e_X[1]
    e_psi = e_X[2]

    # Calculate tangent vector components of R in sph. basis at X
    v_theta = np.dot(R, e_theta)
    v_psi = np.dot(R, e_psi)
    
    # Calculate and normalise tangent vetor
    T = v_theta * e_theta + v_psi * e_psi
    t = T/np.linalg.norm(T)

    return t

def goodman(N, a):
    """
    Obtains random samples from the Goodman probability
    distribution (used for emcee methods) to ensure
    symmetry of "stretch" algorithm.

    Parameters
    ---------- 
    a: float
        Scaling parameter for distribution
    N: int
        Number of iterations carried out by sampling procedure.

    Returns
    -------    
    numpy array
        array of random samples from Goodman distribution
    """
    assert (a > 1), "'a' must be greater than 1."

    return (((a - 1) * np.random.rand(N) + 1) ** 2 / a);

def acor_calc(X):
    """
    """
    D = len(X[0])
    taus = []

    for i in range(D):
        tau = acor.acor(X[:,i])
        taus.append(tau)

    taus = np.array(taus)
    print('Min autocorr time:',np.amin(taus))
    print('Max autocorr time:',np.amax(taus))
    print('Median autocorr time:', np.median(taus))
    print('Mean autocorr time:', np.mean(taus))

    return taus

