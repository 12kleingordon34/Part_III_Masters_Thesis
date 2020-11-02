import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous
import tqdm

def average(arr, n):
    """ Calculates the average of every n points in the given array arr.
        Calculates maximum length of array arr tthat can be averaged over n points
        (variable 'max_div'). Reshapes the array and takes an average over n.

        Any leftover points that are not included in arr[:max_div] are ignored.

        Parameters
        ----------
        arr: numpy array 
            The array to be averaged over.
        n: int
            The number of elements in the array to be averaged over.

        Returns
        _______
        av_arr: numpy array 
            original input 'arr' arragy averaged over every n elements
        length: integer
            length of 'arr' that was used for for the averaging process
    """
    # Find maximum size of array that is divisible by n
    max_div =  n * int(len(arr)/n)

    # Average 'arr' over every n points
    av_arr = np.mean(arr[:max_div].reshape(-1, n), 1)
    length = len(arr[:max_div])
    return av_arr, length

def rms(mean_samples):
    """ Calculates the mean walker rms velocity and displacement from the origin
        
        Parameters
        ----------
        mean_samples: numpy array
            Array of mean walker positions following MH sampling process

        Returns
        -------
        x_rms: numpy array
            Array of mean walker RMS displacements from the origin
        v_rms: numpy array
            Array of mean walker RMS velocities
        v_rms_av: numpy array
            Array of mean walker RMS velocities obtained from v_rms,
            Obtained by averaging v_rms over n successive points 

    """
    # Calculate mean square velocities of mean walker at every step
    dx = 1.
    vel = np.diff(mean_samples, axis=0)/dx    # Array of mean walker velocity vectors
    v_rms = np.average(np.sqrt(vel**2), axis=1) 
    
    # Average v_rms over n=len(v_rms)/50. to reduce effect of fluctuations
    n = int(len(v_rms)/100)
    v_rms_av, length = average(v_rms, n)
    
    #Calculate walker average rms displacement
    x_rms = np.average(np.sqrt(mean_samples**2), axis=1)  
    return x_rms, v_rms, v_rms_av, n, length 

def mean_walker_plot(mean_samples, filename):
    """ Plots the RMS velocites and displacements of the mean walker position.
        Outputs plot in .pdf format.

        Parameters
        ----------
        mean_samples: numpy array
            Positions of mean walker
        filename: str
                
        Returns
        -------
        Plot saved under "'filename'.pdf"

    """

    # Find rms values using function 'rms()'
    x_rms, v_rms, v_rms_av, n, length = rms(mean_samples)
    t = np.arange(len(x_rms))
    
    # Plot figure
    fig2, ax2 = plt.subplots()
    ax2.set_ylim([0, np.amax(v_rms)])
    ax2.set_xlim([0, 10000])
    ax2.plot(t[1:], v_rms, 'b', label='Vrms')
    ax2.plot(t[0:length:n], v_rms_av, 'k', label='Averaged Vrms')
    ax2.set_ylabel('Average RMS velocity (arbitrary units)', color='b')
    ax2.set_xlabel('Sample Number')
    ax2.tick_params('y', colors='b')
    ax2.legend()
    
    # Set axis for mean sq displacement
    ax3 = ax2.twinx()
    ax3.set_ylim([0, np.amax(x_rms)])
    ax3.set_xlim([0, np.amax(t)])
    ax3.plot(t, x_rms, 'r', label='Xrms')
    ax3.set_ylabel('Average RMS displacement from origin', color='r')
    ax3.tick_params('y', colors='r')
    output = filename+'.pdf'
    fig2.savefig(output)
    return

def sub_chain(samples, stepper, S):
    """ Calculates the mean value of each individual walker in the ensemble.

        Parameters
        ----------


        Returns
        -------

    """
    walker_mean = [] 

    for i in range(S):
        i_index, = np.where(stepper == i)
        i_mean = np.average(samples[i_index]) 
        walker_mean.append(i_mean)

    return np.array(walker_mean)


def lhood_calls(burnt_samples, burnt_weights, true_mean, sample_error):
    """ Calculates the number of likelihood calls required to estimate
        the mean to a given sample error

        Parameters
        ----------
        burnt_samples: numpy array
            MH samples with burn in period removed 

        burnt_weights: numpy array
            MH weights with burn in period removed 

        true_mean: numpy array
            The true mean of the sample distribution

        sample_error: float
            The absolute sample error of the MH process

        Returns
        -------
        temp_like.pdf: Figure
            Plot showing the estimated mean as a function of likelihood
            iterations

        max_calls: int
            Maximum number of likelihood calls for all future estimates
            to lie within the sample_error

    """
    # Calculate cumulative mean estimates after burn in period
    X = np.cumsum(burnt_samples, axis=1).T
    Y = np.arange(len(burnt_samples)) + 1
    Z = X/Y
    
    # Calculate difference of mean estimates to the true mean
    delta = np.absolute(Z.T - true_mean)
    deviation = np.sqrt(np.sum(delta**2, axis=1))     # Calculate difference of calculated to true mean 

    # Cumulative number of likelihood calls. The ith element 
    # corresponds to the number of samples to obtain the ith mean 
    # estimate in array 'Z'
    N = np.cumsum(burnt_weights)

    # Plot estimated mean against number of likelihood calls
    fig, ax = plt.subplots()
    ax.plot(N, deviation)
    ax.plot(N, [sample_error]*len(N))
    ax.set_xlim([0,1000])
    ax.set_ylim([0,2])
    plt.xlabel('Number of likelihood Calls')
    plt.ylabel('Absolute Deviation')
    plt.title('Deviation of sample and true means vs. # of Likelihood Calls')
    plt.savefig('temp_like.pdf')

    # Find indexes for mean estimates that are greater than the sample error
    indexes = np.where(deviation > sample_error)

    # Find maximum index of array 'indexes'
    max_index = np.amax(indexes)
    max_calls = N[max_index]
    return max_calls

def sph_tester(logP, m, n):
    theta = np.linspace(0, np.pi, m)
    phi = np.linspace(0, 2 * np.pi, m)

    xy = np.mgrid[0:(2*np.pi):(2*np.pi)/m, 0:(np.pi):(np.pi)/m].reshape(2,-1).T
    prob = np.exp(np.array([logP(xy[i]) for i in tqdm.tqdm(range(m**2-1))]))
    norm_prob = prob/np.sum(prob)

    a = np.random.choice(np.arange(0, m**2-1), size=n, p=norm_prob)
    return xy[a]

def matrix_stats(M):
    '''
    Calculates the mean and standard deviation of on and off diagonal 
    matrix elements for a covariance matrix

    Parameters
    ----------
    M: numpy array
        Covariance Matrix

    Returns
    -------
    mean_on: float
        average of diagonal matrix elements
    std_on: float
        std of of diagonal matrix elements
    mean_off: float
        average of of off-diagonal matrix elements
    std_off: float
        std of of off-diagonal matrix elements
    '''
    off_diag = np.extract(1-np.eye(len(M)), M)
    diag = np.extract(np.eye(len(M)), M)

    mean_off = np.mean(off_diag)
    std_off = np.std(off_diag)
    mean_on = np.mean(diag)
    std_on = np.std(diag)

    return mean_on, std_on, mean_off, std_off

def vector_stats(V):
    '''
    Calculates the mean and standard deviation of elements in a 1D array

    Parameters
    ----------
    V: numpy array
        Vector

    Returns
    -------
    mean: float
        average of vector elements
    std: float
        std of of vector elements
    '''

    return np.mean(V), np.std(V)

def total_samples(samples, weights):
    '''
    Calculates a list of all accepted and rejected trial steps
    from a set of samples and their weights

    Parameters
    ----------
    samples: numpy array
    weights: numpy array

    Returns
    -------
    total_samples: numpy array
    ''' 
    assert len(samples) == len(weights)

    size = len(samples[0])
    total_samples = np.ones((1,size))
    
    for i in range(len(samples)):
        temp = np.repeat([samples[i]], weights[i], axis=0)
        total_samples = np.append(total_samples,temp,axis=0)
    
    total_samples = np.delete(total_samples, 0, 0)

    return total_samples

def acor(chain):
    rhos = []
    c = chain - chain.mean()
    var = np.einsum('ij,ij->i',c,c)

    for i in tqdm.tqdm(range(1,len(c))):
        rho = np.einsum('ij,ij->i',c[:-i],c[i:])/var
        if rho < 0:
            break
        rhos.append(abs(rho))
    print(len(rhos))

    return 1 + 2*sum(rhos)

def acor_v2(chain):
    C_t = []
    c = chain - chain.mean()
    var = np.einsum('ij,ij->i',c,c)
    C_0 = np.sum(np.multiply(c,c)/len(c))

    for i in tqdm.tqdm(range(1,len(c))):
        C_i = np.sum(np.multiply(c[:-i], c[i:])/len(c[i:]))
        if C_i < 0:
            break
        C_t.append(C_i)
    print(len(C_t))

    return (C_0 + sum(C_t))/C_0
