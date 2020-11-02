import matplotlib.pyplot as plt
import numpy as np
from numpy import pi
from ligo_likelihood import source, timedata as ts, ligo_loglikelihood as l, m0
from metropolis_hastings import distributions as dists
import ligo_likelihood
from metropolis_hastings.samplers import refined as run_mh_ref, emcee
from metropolis_hastings import tools as st
from metropolis_hastings import stepper as step
import getdist, IPython
import misc as ms
from getdist import plots, MCSamples
import acor
import time
import random

# Set random seed
q=1
random.seed(q)
np.random.seed(q)


# Test bimodal multivariate gaussian likelihood
def logP_2(x):
    """ Set covariance and mean of gaussian symmetric in theta """
    Mu = np.array([2*np.pi, np.pi/2])
    Mu1 = np.array([0, np.pi/2])
    Sigma = np.array([[0.4, 0],[0, 0.3]])

    phi = x[0]
    theta = x[1]
    if x[0] < 0 or x[1] < 0 or x[0] > 2*pi or x[1] > pi:
        return float('-inf')
    return dists.bimodal_multi_gaussian(x, Mu, Sigma, Mu1, Sigma)

 
# Set parameters for sampling process
D = 2                            # Dimension of the system
S = 150                           # Number of walkers
N = 500000                         # Number of iterations
a = 2.0                          # emcee scaling parameter 


print('seed:',q)
print('N:',N)
print('S:', S)
print('(a):','(',a,')')


#Setting initial walker positions
x_0 = np.random.uniform(size=(S,D)) * (2*pi,pi)
alpha = 1.                     # Scaling factor for trial step

# Set ADS sampler as function U
def U(walkers):
    """ Calculate the trial step used for walker """
    return step.ads(walkers,alpha=alpha) 

# Spherical stepping function V (Euclidian approximation)
def EA(walkers):
    """ Calculate the trial step for walker using spherical
        geometry """
    return step.sph_euc_approx(walkers,alpha=alpha)

# Spherical stepping function W (Parallel Transport Method)
def PT(walkers):
    """ Calculate the trial step for walker using spherical
        geometry """
    return step.sph_parallel_transport(walkers,beta=1.)

# Spherical emcee stepping function X
def X(walkers,beta):
    """ Calculate trial step for walker in spherical geometry
        using emcee method, where beta is the scaling factor
        determined by Goodman."""
    return step.sph_emcee(walkers,beta=a)

# Euclidean emcee stepping function Y
def Y(walkers,beta):
    """ Calculate trial step for walker in spherical geometry
        using emcee method, where beta is the scaling factor
        determined by Goodman."""
    return step.emcee(walkers,beta=a)

# Obtain mean and covaraince drawn from the distribution
m = 30
test = ms.sph_tester(logP_2, m, 10*N)
print('Theoretical Mean:','\n', np.average(test.T, axis=1))
print('Theoretical Covariance:','\n', np.cov(test.T))

# Run ADS PT sampler using parallel transport stepping method
samples, weights, mean_samples, walk_identity, tot_samp = run_mh_ref(logP_2, PT, x_0, N)
print('P.T. Sample Size: ', len(samples),',','Acc. Ratio:',len(samples)/len(tot_samp))
start = time.time()
acor.acor(mean_samples.T)
taus1 = st.acor_calc(mean_samples)
print("Process time/seconds: ", time.time()-start)

# Run ADS sampler using euclidean stepping method
x_0 = np.random.uniform(size=(S,D)) * (2*pi,pi)
samples1, weights1, mean_samples1, walk_identity1, tot_samp1 = run_mh_ref(logP_2, EA, x_0, N)
print('Euc. Approx Sample Size: ', len(samples1),',','Acc. Ratio:',len(samples1)/len(tot_samp1))
start = time.time()
print(acor.acor(mean_samples1.T))
taus1 = st.acor_calc(mean_samples1)
print("Process time/seconds: ", time.time()-start)


# Run emcee spherical stepping method
x_0 = np.random.uniform(size=(S,D)) * (2*pi,pi)
samples2, weights2, mean_samples2, walk_identity2, tot_samp2 = emcee(logP_2, X, x_0, N, a)
print('Sph. emcee Sample Size: ', len(samples2),',','Acc. Ratio:',len(samples2)/len(tot_samp2))
start = time.time()
print(acor.acor(mean_samples2.T))
taus2 = st.acor_calc(mean_samples2)
print("Process time/seconds: ", time.time()-start)

# Run Euclidean ADS sampler 
samples3, weights3, mean_samples3, walk_identity3, tot_samp3 = run_mh_ref(logP_2, U, x_0, N)
print('Euclidean ADS Sample Size: ', len(samples3),',','Acc. Ratio:',len(samples3)/len(tot_samp3))
start = time.time()
print(acor.acor(tot_samp3.T))
taus2 = st.acor_calc(mean_samples3)
print("Process time/seconds: ", time.time()-start)

# Run emcee Euclidean stepping method
x_0 = np.random.uniform(size=(S,D)) * (2*pi,pi)
samples4, weights4, mean_samples4, walk_identity4, tot_samp4 = emcee(logP_2, Y, x_0, N, a)
print('Euclidean. emcee Sample Size: ', len(samples4),',','Acc. Ratio:',len(samples4)/len(tot_samp4))
start = time.time()
print(acor.acor(mean_samples4.T))
taus4 = st.acor_calc(mean_samples4)
print("Process time/seconds: ", time.time()-start)


# Configure Triangle plots using 'getdist'
frac = 0.10     # Choose fraction of data points to ignore
b_ind = int(len(samples)*frac)
names = ["\phi","\\theta"]
labels =  ["\phi","\\theta"]
sph_samples = MCSamples(samples=samples, names=names, labels=labels)
euc_samples = MCSamples(samples=samples1[b_ind:], names=names, labels=labels)
emcee_samples = MCSamples(samples=samples2[b_ind:], names=names, labels=labels)
ADSeuc_samples = MCSamples(samples=samples3[b_ind:], names=names, labels=labels)
emceeeuc_samples = MCSamples(samples=samples4, names=names, labels=labels)
true_samples = MCSamples(samples=test[:], names=names, labels=labels)

# Set Latex font for figures
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Histogram of walker weights
fig1, ax1 = plt.subplots()
bins = np.arange(1,9) - 0.5
ax1.hist([weights, weights1, weights2, weights3, weights4], bins, normed=True, alpha=0.5, label=['PTSS','EASS','SE','Euc. ADS','Euc. emcee'])
ax1.set_xlabel('Sample Weights')
ax1.set_ylabel('Normalised Walker Density')
ax1.set_xlim([0.5,5.5])
#plt.title(r'Histogram of Walker Weights (Sph)', fontsize=16)
plt.legend(loc='upper right')
fig1.savefig('hist_sph.pdf')

# Print weighted correlation and mean of walker ensemble (Parallel Transport Method)
w_average = np.average(samples[b_ind:], weights=weights[b_ind:], axis=0)
print('Weighted (P.T.) Sample Mean:','\n', w_average)
w_covariance = np.cov(samples[b_ind:], aweights=weights[b_ind:], rowvar=0) 
print('Weighted (P.T.) Sample Covariance:','\n',w_covariance)

# Print weighted correlation and mean of walker ensemble (Spherical Euclidean approx)
w_average1 = np.average(samples1[b_ind:], weights=weights1[b_ind:], axis=0)
print('Weighted (Euclidean Approx.) Sample Mean:','\n', w_average1)
w_covariance1 = np.cov(samples1[b_ind:], aweights=weights1[b_ind:], rowvar=0) 
print('Weighted (Euclidean Approx.) Sample Covariance:','\n',w_covariance1)

# Print weighted correlation and mean of walker ensemble (emcee)
w_average2 = np.average(samples2[b_ind:], weights=weights2[b_ind:], axis=0)
print('Weighted (Spherical emcee) Sample Mean:','\n', w_average2)
w_covariance2 = np.cov(samples2[b_ind:], aweights=weights2[b_ind:], rowvar=0) 
print('Weighted (Spherical emcee) Sample Covariance:','\n',w_covariance2)

# Print weighted correlation and mean of walker ensemble (ADS)
w_average3 = np.average(samples3[b_ind:], weights=weights3[b_ind:], axis=0)
print('Weighted (Euc ADS) Sample Mean:','\n', w_average3)
w_covariance3 = np.cov(samples3[b_ind:], aweights=weights3[b_ind:], rowvar=0) 
print('Weighted (Euc ADS) Sample Covariance:','\n',w_covariance3)

# Print weighted correlation and mean of walker ensemble (ADS)
w_average4 = np.average(samples4[b_ind:], weights=weights4[b_ind:], axis=0)
print('Weighted (Euc emcee) Sample Mean:','\n', w_average4)
w_covariance4 = np.cov(samples4[b_ind:], aweights=weights4[b_ind:], rowvar=0) 
print('Weighted (Euc emcee) Sample Covariance:','\n',w_covariance4)

# Generate GetDist Plots
g = plots.getSubplotPlotter()
g.triangle_plot([sph_samples, emceeeuc_samples], filled=True, colors=['red','blue'], legend_labels=['PTSS','Euc. emcee'], legend_loc='upper right')
g.export('triangle_sph.pdf')
