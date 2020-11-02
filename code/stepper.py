''' Set of functions used to calculate the stepping direction.
'''

import numpy as np
import random
from metropolis_hastings import tools
from scipy.linalg import expm
#import tools

def ads(walkers, alpha):
    '''
    Calculates the stepping vector for the trial walker

    Parameters
    ----------
    walkers: numpy array
        Positions of walker ensemble
    alpha: float
        Scaling factor for step size

    Returns
    -------
    R[0]: int
        Ensemble index of trial walker
    alpha*(A - B): numpy array
        Trial step
    '''
    # Randomly select three walkers from ensemble
    S = len(walkers)
    R = random.sample(range(S), 3)
    _, A, B = walkers[R]

    return R[0], alpha*(A - B)

def sph_parallel_transport(walkers, beta):
    '''
    Generates a trial step (using the emcee method proposed by
    Goodman) on the surface of a sphere.

    Parameters
    ----------
    walkers: numpy array
        Positions of walker ensemble
    beta: float
        Scaling factor for Goodman scaling distribution

    Returns
    -------
    j: int
        Ensemble index of trial walker
    (X_trial-X): numpy array
        Trial step
    '''

    # Randomly select 2 unique walkers from ensemble
    S = len(walkers)
    R = random.sample(range(S), 3)   # Walkers must be unique
    j = np.copy(R[0])
    
    X, A, B = walkers[R]

    # Spherical 3D basis set for points X, A, B
    e_X = tools.sph_basis(X) 
    e_A = tools.sph_basis(A) 
    e_B = tools.sph_basis(B) 

    # Find angle between A and B, A and X
    angle_xa = np.arccos(np.dot(e_X[0,:],e_A[0,:]))
    angle_ab = np.arccos(np.dot(e_B[0,:],e_A[0,:]))

    # Cartesian vector connecting points A to B
    R_ab = e_A[0] - e_B[0]

    # Calculate tangent vector connecting A to B
    t_ab = tools.sph_tangent(R_ab,e_A)

    # Find axis of great circle defined by OA, OX 
    n_GS1 = np.cross(e_A[0], e_X[0])
    n_x1 = np.cross(np.identity(3), n_GS1)

    # Rotate t_ab from A to X
    R = expm(angle_xa * n_x1)
    t_pt = R.dot(t_ab)
   
    # Find axis of normal to great circle defined by (t_pt x R_X)
    n_GS2 = np.cross(e_X[0,:], t_pt)
    n_x2 = np.cross(np.identity(3), n_GS2)

    # Find trial point by rotating about GS2
    R = expm(beta * angle_ab * n_x2)
    X_trial_cart = R.dot(e_X[0,:])
    X_trial = tools.car2sph(X_trial_cart)

    return j, np.array((X_trial-X))

def sph_euc_approx(walkers, alpha):
    '''
    Calculates step direction by assuming local euclidean geometry
    for all 3 selected walkers
    i.e. v (prop) angle_xa * t_xa + angle_xb * t_xb

    Parameters
    ----------
    walkers: numpy array
        Positions of walker ensemble
    alpha: float
        Scaling factor for step size

    Returns
    -------
    j: int
        Ensemble index of trial walker
    (X_trial-X): numpy array
        Trial step
    '''

    # Randomly select 3 unique walkers from ensemble
    S = len(walkers)
    R = random.sample(range(S), 3)   # Walkers must be unique
    j = np.copy(R[0])
    
    X, A, B = walkers[R]

    # Spherical 3D basis set for points X, A, B
    e_X = tools.sph_basis(X) 
    e_A = tools.sph_basis(A) 
    e_B = tools.sph_basis(B) 

    # Find angle between A and B, X and A, X and B
    angle_ab = np.arccos(np.dot(e_A[0,:],e_B[0,:]))
    angle_xa = np.arccos(np.dot(e_X[0,:],e_A[0,:]))
    angle_xb = np.arccos(np.dot(e_X[0,:],e_B[0,:]))

    # Cartesian vector connecting points X to A and X to B
    R_xa = e_A[0,:] - e_X[0,:]
    R_xb = e_B[0,:] - e_X[0,:]

    # Calculate tangent vector connecting X to A
    t_xa = tools.sph_tangent(R_xa,e_X)
   
    # Calculate tangent vector connecting X to A
    t_xb = tools.sph_tangent(R_xb,e_X)
    
    # Find normalised vector pointing in step's direction tangent to the sphere at X
    step = angle_xa * t_xa + angle_xb * t_xb
    v_step = step/np.linalg.norm(step)

    # Find axis of normal to great circle defined by (v_step x R_X)
    n_GS = np.cross(e_X[0,:], v_step)
    n_x = np.cross(np.identity(3), n_GS)

    # Find trial point
    R = expm(alpha * angle_ab * n_x)
    X_trial_cart = R.dot(e_X[0,:])
    X_trial = tools.car2sph(X_trial_cart)

    return j, np.array((X_trial-X))

def sph_leapfrog(walkers, alpha):
    '''
    Calculates step direction using "stretch" algorithm similar to 
    euclidean form. Finds tangent between two points and "leapfrogs"
    to obtain trial point
    i.e. v (prop) alpha *angle_xa * t_xa

    Parameters
    ----------
    walkers: numpy array
        Positions of walker ensemble
    alpha: float
        Scaling factor for step size

    Returns
    -------
    j: int
        Ensemble index of trial walker
    (X_trial-X): numpy array
        Trial step
    '''

    # Randomly select 2 unique walkers from ensemble
    S = len(walkers)
    R = random.sample(range(S), 2)   # Walkers must be unique
    j = np.copy(R[0])
    
    X, A = walkers[R]

    # Spherical 3D basis set for points X, A, B
    e_X = tools.sph_basis(X) 
    e_A = tools.sph_basis(A) 

    # Find angle between A and B, X and A, X and B
    angle_xa = np.arccos(np.dot(e_X[0,:],e_A[0,:]))

    # Cartesian vector connecting points X to A and X to B
    R_xa = e_A[0,:] - e_X[0,:]

    # Calculate tangent vector connecting X to A
    t_xa = tools.sph_tangent(R_xa,e_X)
   
    # Find normalised vector pointing in step's direction tangent to the sphere at X
    step = angle_xa * t_xa
    v_step = step/np.linalg.norm(step)

    # Find axis of normal to great circle defined by (v_step x R_X)
    n_GS = np.cross(e_X[0,:], v_step)
    n_x = np.cross(np.identity(3), n_GS)

    # Find trial point
    R = expm(2. * alpha * angle_xa * n_x)
    X_trial_cart = R.dot(e_X[0,:])
    X_trial = tools.car2sph(X_trial_cart)

    return j, np.array((X_trial-X))

def sph_emcee(walkers, beta):
    '''
    Generates a trial step (using the emcee method proposed by
    Goodman) on the surface of a sphere.

    Parameters
    ----------
    walkers: numpy array
        Positions of walker ensemble
    beta: float
        Scaling factor for Goodman scaling distribution

    Returns
    -------
    j: int
        Ensemble index of trial walker
    (X_trial-X): numpy array
        Trial step
    '''

    # Randomly select 2 unique walkers from ensemble
    S = len(walkers)
    R = random.sample(range(S), 2)   # Walkers must be unique
    j = np.copy(R[0])
    
    X, A = walkers[R]

    # Spherical 3D basis set for points X, A, B
    e_X = tools.sph_basis(X) 
    e_A = tools.sph_basis(A) 

    # Find angle between A and B, X and A, X and B
    angle_xa = np.arccos(np.dot(e_X[0,:],e_A[0,:]))

    # Cartesian vector connecting points X to A and X to B
    R_xa = e_A[0,:] - e_X[0,:]

    # Calculate tangent vector connecting X to A
    t_xa = tools.sph_tangent(R_xa,e_X)
   
    # Find normalised vector pointing in step's direction tangent to the sphere at X
    step = angle_xa * t_xa
    v_step = step/np.linalg.norm(step)

    # Find axis of normal to great circle defined by (v_step x R_X)
    n_GS = np.cross(e_X[0,:], v_step)
    n_x = np.cross(np.identity(3), n_GS)

    # Find trial point
    R = expm((1-beta) * angle_xa * n_x)
    X_trial_cart = R.dot(e_X[0,:])
    X_trial = tools.car2sph(X_trial_cart)

    return j, np.array((X_trial-X))

def toroidal_ads(walkers, alpha):
    '''
    Calculates the stepping vector for the trial walker
    for an angular toroidal sampling space.

    Sampling space period is 2*pi

    Parameters
    ----------
    walkers: numpy array
        Positions of walker ensemble
    alpha: float
        Scaling factor for step size

    Returns
    -------
    R[0]: int
        Ensemble index of trial walker
    alpha*(A - B): numpy array
        Trial step
    '''
    # Randomly select three walkers from ensemble
    S = len(walkers)
    R = random.sample(range(S), 3)
    X, A, B = walkers[R]
    N = len(X)
    step = np.empty(N)

    # Obtain toroidal sampling step
    c = 2 * np.pi
    z1 = (A-B) % c
    z2 = (B-A) % c

    for i in range(N):
        if z1[i] <= z2[i]:
            step[i] = z1[i]
        else:
            step[i] = -z2[i]

    # Find trial point
    X_trial = (X+step) % c

    return R[0], np.array((X_trial-X))

def toroidal_emcee(walkers, beta):
    '''
    Calculates the stepping vector for the trial walker
    for an angular toroidal sampling space.

    Sampling space period is 2*pi

    Parameters
    ----------
    walkers: numpy array
        Positions of walker ensemble
    beta: float
        Scaling factor for Goodman scaling distribution

    Returns
    -------
    R[0]: int
        Ensemble index of trial walker
    (X_trial-X): numpy array
        Trial step
    '''
    # Randomly select three walkers from ensemble
    S = len(walkers)
    R = random.sample(range(S), 3)
    X, A, B = walkers[R]
    N = len(X)
    step = np.empty(N)

    # Obtain toroidal sampling step
    c = 2 * np.pi
    z1 = (A-B) % c
    z2 = (B-A) % c

    for i in range(N):
        if z1[i] <= z2[i]:
            step[i] = z1[i]
        else:
            step[i] = -z2[i]

    # Find trial point
    X_trial = (X+step*(1-beta)) % c

    return R[0], np.array((X_trial-X))

def emcee(walkers, beta):
    '''
    Calculates the stepping vector for the trial walker

    Parameters
    ----------
    walkers: numpy array
        Positions of walker ensemble
    beta: float
        Scaling factor for Goodman scaling distribution

    Returns
    -------
    R[0]: int
        Ensemble index of trial walker
    (X_trial-X): numpy array
        Trial step
    '''
    # Randomly select three walkers from ensemble
    S = len(walkers)
    R = random.sample(range(S), 2)
    X, A = walkers[R]

    return R[0], (1-beta)*(A-X)

def multidim_emcee(walkers, beta, num):
    '''
    Calculates the stepping vector for the trial walker

    Parameters
    ----------
    walkers: numpy array
        Positions of walker ensemble
    W: int
        Number of walkers used to obtain trial step
    beta: float
        Scaling factor for Goodman scaling distribution

    Returns
    -------
    R[0]: int
        Ensemble index of trial walker
    (X_trial-X): numpy array
        Trial step
    '''
    # Randomly select three walkers from ensemble
    S = len(walkers)  #len(walkers[0])
    R = random.sample(range(S), num)
    X = np.copy(walkers[R[0]])
    Y = walkers[R[1:]]
    Z = Y-X
    Z = np.sum(Z,axis=0)

    return R[0], (1-beta) * Z/(num-1)
