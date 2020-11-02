''' Functions used for calculating steps in curvelinear coordinates.
'''

import numpy as np

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

    theta = x[0]
    psi = x[1]

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

    basis = [r, e_the, e_psi]
    
    return np.array(basis)

def car2sph(x):
    '''
    Convert cartesian coordinates to spherical angles, theta and psi.

    Parameters
    ---------- 
    x: numpy array
        asdf

    Returns
    -------
    sph_coord: numpy array
        asdf

    '''
    sph_coord = np.empty(2)
    xy = x[0] **2 + x[1] ** 2
    sph_coord[0] = np.arcsin(np.sqrt(xy)) # for elevation angle defined from Z-axis down
    sph_coord[1] = np.arctan2(x[1], x[0]) % (2 * np.pi) # for elevation angle defined from XY-plane up

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

    # Calculate tangent vector components of R in sph. basis at X
    #print(R)
    v_theta = np.dot(R, e_X[1,:])
    v_psi = np.dot(R, e_X[2,:])
    
    # Calculate and normalise tangent vetor
    T = v_theta * e_X[1,:] + v_psi * e_X[2,:] 
    t = T/np.linalg.norm(T)

    return t
