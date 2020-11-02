""" Metropolis hastings sampling procedures.
"""
import numpy as np
import tqdm
import matplotlib.pyplot as plt
from IPython import display
import time
from metropolis_hastings import tools 


def basic(logP, Q, x_0, n):
    """ Runs a simple Metropolis Hastings sampling procedure

        Parameters
        ----------
        P: function
            The distribution we're sampling.
            Examples can be found in:
                functions.py
        Q: function
            The proposal distribution Q(x_0)
            Examples can be found in:
            metropolis_hastings/proposal_distributions.py
        x_0: numpy.array
            Initial start point
        n: int
            stopping criterion (# of iterations)

        Returns
        -------
        A numpy array of samples from the distribution P
    """
    samples = []
    weights = []
    walker_weights = [1]

    for _ in tqdm.tqdm(range(n)):
        x_1, q_rat = Q(x_0)  # Propose new point

        acceptance_ratio = np.exp(logP(x_1) - logP(x_0)) * q_rat

        # Determine next sample step value
        if acceptance_ratio > np.random.rand():
            x_0 = x_1
            weights.append(walker_weights)
            walker_weights = [1]
        else:                                          # If step is rejected
            walker_weights[0] += 1

        # Append the new point to the array
        samples.append(x_0) 

    return np.array(samples), np.array(weights)
    

def refined(logP, U, walkers, n):
    """ Runs a refined sampling procedure using ensemble step calculations

        Parameters
        ----------
        logP: function
            The distribution we're sampling. Probabilities given in log likelihoods.
            Examples can be found in:
                functions.py
        U: function
            The stepping process we're using.
            Examples can be found in:
                metropolis_hastings/stepper.py
        walkers: numpy.array
            Initial start point of walker ensemble
        n: int
            stopping criterion (# of iterations)

        Returns
        -------
        return np.array(samples), np.array(weights), np.array(mean_samples), np.array(stepper)

        samples: numpy array
             Samples from the distribution P obtained by the sampler ensemble.
        weights: numpy array
             The weights corresponding to the steps registered in array 'samples'
        mean_samples: numpy array
             The mean walker positions after every successful step
        identity: numpy array
             The label of the specific walker which made a step
    """
   
    # Calculate mean position of walkers
    mean_walker = np.average(walkers, axis=0)
    D = len(mean_walker)
    S = len(walkers)

    # Figures for dynamic plot
    plt.ion()                                                 # For dynamic plotting
    fig, ax = plt.subplots()
    xmin, xmax = min(walkers[:,0]),max(walkers[:,0])
    ymin, ymax = min(walkers[:,1]),max(walkers[:,1])
    xmin, xmax = 0, 2 * np.pi
    ymin, ymax = 0, 2* np.pi
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if D == 2:
        x = np.linspace(xmin,xmax,20)
        y = np.linspace(ymin,ymax,20)
        z = np.array([[logP([xi,yi]) for xi in x] for yi in tqdm.tqdm(y)])
        CT = ax.contour(x,y,z)
        ax.clabel(CT, inline=1, fontsize=5, inline_spacing=3)
    dyn_ax, = ax.plot(walkers[:,0], walkers[:,1], 'ko')        # For dynamic plotting
    dyn_mean, = ax.plot(mean_walker[0], mean_walker[1], 'ro')

    # Create empty lists to store output of MH process
    samples = []
    mean_samples = []
    weights = []
    walker_weights = [1] * S
    corr = []
    identity = []
    total_samples = []

    for i in tqdm.tqdm(range(n)):

        # Select trial step
        j, Z = U(walkers)
        X = np.copy(walkers[j])

        # Propose new point X_new for walker 'x'
        X_new = X + Z
        acceptance_ratio = np.exp(logP(X_new) - logP(X))

        # Determine next sample step value
        if acceptance_ratio > np.random.rand():         # If step is accepted
            samples.append(X)
            weights.append(walker_weights[j])

            walkers[j] = X_new[:]
            mean_walker = np.average(walkers, axis=0)
            mean_samples.append(mean_walker)
            walker_weights[j] = 1
            identity.append(j)
        else:                                          # If step is rejected
            walker_weights[j] += 1

        # Store total set of trial points
        total_samples.append(X)
        mean_walker = np.average(walkers, axis=0)
        mean_samples.append(mean_walker)
        

        ## Dynamic plot of first three walkers    ## For dynamic plotting 
        #if i%S == 0:                            ## For dynamic plotting
        #    dyn_ax.set_xdata(walkers[:,0]) ## For dynamic plotting 
        #    dyn_ax.set_ydata(walkers[:,1]) ## For dynamic plotting
        #    dyn_mean.set_xdata(mean_walker[0])
        #    dyn_mean.set_ydata(mean_walker[1])
        #    fig.canvas.draw()    ## For dynamic plotting
        #    plt.pause(0.000000000001)

    return np.array(samples), np.array(weights), np.array(mean_samples), np.array(identity), np.array(total_samples)


def emcee(logP, U, walkers, n, a):
    """ Runs emcee sampling procedure using ensemble step calculations

        Parameters
        ----------
        logP: function
            The distribution we're sampling. Probabilities given in log likelihoods.
            Examples can be found in:
                functions.py
        U: function
            The emcee stepping process we're using.
            Examples can be found in:
                metropolis_hastings/stepper.py
            (Choose from emcee or sph_emcee)
        walkers: numpy.array
            Initial start point of walker ensemble
        n: int
            stopping criterion (# of iterations)
        a: float
            scaling factor for Goodman distribution

        Returns
        -------
        return np.array(samples), np.array(weights), np.array(mean_samples), np.array(stepper)

        samples: numpy array
             Samples from the distribution P obtained by the sampler ensemble.
        weights: numpy array
             The weights corresponding to the steps registered in array 'samples'
        mean_samples: numpy array
             The mean walker positions after every successful step
        identity: numpy array
             The label of the specific walker which made a step
    """
    # Generate random numbers from Goodman distribution
    rand_arr = tools.goodman(n, a)
   
    # Calculate mean position of walkers
    mean_walker = np.average(walkers, axis=0)
    D = len(mean_walker)
    S = len(walkers)

    # Figures for dynamic plot
    plt.ion()                                                 # For dynamic plotting
    fig, ax = plt.subplots()
    xmin, xmax = min(walkers[:,0]),max(walkers[:,0])
    ymin, ymax = min(walkers[:,1]),max(walkers[:,1])
    xmin, xmax = -0.1, 2.1 * np.pi
    ymin, ymax = -0.1, 2.1 * np.pi
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if D == 2:
        x = np.linspace(xmin,xmax,20)
        y = np.linspace(ymin,ymax,20)
        z = np.array([[logP([xi,yi]) for xi in x] for yi in tqdm.tqdm(y)])
        CT = ax.contour(x,y,z)
        ax.clabel(CT, inline=1, fontsize=5, inline_spacing=3)
    dyn_ax, = ax.plot(walkers[:,0], walkers[:,1], 'ko')        # For dynamic plotting
    dyn_mean, = ax.plot(mean_walker[0], mean_walker[1], 'ro')

    # Create empty lists to store output of MH process
    samples = []
    mean_samples = []
    weights = []
    walker_weights = [1] * S
    identity = []
    total_samples = []

    for i in tqdm.tqdm(range(n)):

        # Select trial step
        j, Z = U(walkers, rand_arr[i])
        X = np.copy(walkers[j])

        # Propose new point X_new for walker 'x'
        X_new = X + Z
        acceptance_ratio = np.exp(logP(X_new) - logP(X)) * (rand_arr[i] ** (D-1))

        # Determine next sample step value
        if acceptance_ratio > np.random.rand():         # If step is accepted
            samples.append(X)
            weights.append(walker_weights[j])

            walkers[j] = X_new[:]
            walker_weights[j] = 1
            identity.append(j)
        else:                                          # If step is rejected
            walker_weights[j] += 1

        # Store total set of trial points
        total_samples.append(X)
        mean_walker = np.average(walkers, axis=0)
        mean_samples.append(mean_walker)

        ## Dynamic plot of first three walkers    ## For dynamic plotting 
        #if i%S == 0:                            ## For dynamic plotting
        #    dyn_ax.set_xdata(walkers[:,0]) ## For dynamic plotting 
        #    dyn_ax.set_ydata(walkers[:,1]) ## For dynamic plotting
        #    dyn_mean.set_xdata(mean_walker[0])
        #    dyn_mean.set_ydata(mean_walker[1])
        #    fig.canvas.draw()    ## For dynamic plotting
        #    plt.pause(0.00000001)

    return np.array(samples), np.array(weights), np.array(mean_samples), np.array(identity), np.array(total_samples)
