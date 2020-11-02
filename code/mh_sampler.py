""" Python program to run a Metropolis Hastings algorithm.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.mlab as mlab
from metropolis_hastings import distributions as dists
import getdist, IPython
import misc as ms
import random
from metropolis_hastings import tools as st
from metropolis_hastings import stepper as step
from metropolis_hastings.proposal_distributions import gaussian_proposal, multivar_gaussian_proposal
from metropolis_hastings.samplers import basic as run_mh, refined as run_mh_ref, emcee
from getdist import plots, MCSamples
import time
import threading
import acor

# Set random seed
q=2
random.seed(q)
np.random.seed(q)

# General Parameters
D = 2                            # Dimension of the system
S = 150                           # Number of walkers
N = 550000                        # Number of iterations 
q_sig = 1.                  # lengthscale of MH proposal distribution 
x_0 = np.full((S,D), 10.) + 0.2 * np.random.uniform(size=(S,D))    #Setting initial walker positions
x_0emcee = np.copy(x_0)
x_0mh = np.full(D, 10.)
#x_0mh = np.array([10,10,10,10,10])#((1.,D), 10.)
alpha = 1.                      # Scaling factor for trial step
a = 2.                         # Scaling factor for emcee
b = 2.                         # Scaling factor for multidim emcee
print('seed:',q)
print('N:',N)
print('S:', S)
print('(a,b):','(',a,',',b,')')

# Parameters for Multivariate Gaussian
mu=np.zeros(D)
sig_ij = 1.75
sig_ii = 2.
print('(sig_ii, sig_ij):','(',sig_ii,',',sig_ij,')')
Sigma=np.full((D,D), sig_ij) # np.identity(D)
np.fill_diagonal(Sigma, sig_ii)
#Sigma[0,1] = 0.95
#Sigma[1,0] = 0.95


# Correlated Distribution to be sampled logP_0
def logP_0(x):
    """ Set covariance and mean of gaussian """
    return dists.multivariate_gaussian(x, mu, Sigma)

# Test Rosenbrock Likelihood Density
def logP_1(x):
    """ Set covariance and mean of gaussian """
    return dists.Rosenbrock(x)

# Proposal distribution Q
def Q(x):
    """ choose lengthscale for gaussian proposal """
    return gaussian_proposal(x, a=q_sig)

# Proposal Distribution T
def T(x):
    """ choose mean and covariance of multivar gaussian proposal """
    return multivar_gaussian_proposal(x, cov=Sigma/2) 


# Stepping function U
def U(walkers):
    """ Calculate the trial step used for MH walker """
    return step.ads(walkers,alpha=alpha) 

# Stepping function V - emcee
def V(walkers,beta):
    return step.emcee(walkers,beta)

# Stepping function W - multidim emcee
M = D   # Use M walkers to calculate trial step
def Y(walkers, beta): 
    return step.multidim_emcee(walkers,beta,num=M) 

## Run MH refined sampling function, outputing an array of position vectors
samples, weights, mean_samples, walk_identity, tot_samp= run_mh_ref(logP_1, U, np.copy(x_0), N)
print('ADS Sample Size: ', len(samples),',','Acc. Ratio:',len(samples)/len(tot_samp))
start = time.time()
acor.acor(mean_samples.T)
taus1 = st.acor_calc(mean_samples)
print("Process time/seconds: ", time.time()-start)

samples2, weights2, mean_samples2, walk_identity2, tot_samp2 = emcee(logP_1, V, np.copy(x_0), N, a)
print('emcee Sample Size: ', len(samples2),',','Acc. Ratio:',len(samples2)/len(tot_samp2))
start = time.time()
print(acor.acor(mean_samples2.T))
taus2 = st.acor_calc(mean_samples2)
print("Process time/seconds: ", time.time()-start)

samples3, weights3, mean_samples3, walk_identity3, tot_samp3 = emcee(logP_1, Y, np.copy(x_0), N, b)
print('Multi-emcee Sample Size: ', len(samples3),',','Acc. Ratio:',len(samples3)/len(tot_samp3))
start = time.time()
print(acor.acor(tot_samp3.T))
taus2 = st.acor_calc(mean_samples3)
print("Process time/seconds: ", time.time()-start)

samplesMH, weightsMH = run_mh(logP_1, T, x_0mh, N)
print('MH (Calibrated) Sample Size: ', len(samplesMH),',','Acc. Ratio:',len(weightsMH)/len(samplesMH))
start = time.time()
print(acor.acor(samplesMH.T))
tausMH = st.acor_calc(samplesMH)
print("Process time/seconds: ", time.time()-start)


# Proposal Distribution T
def T1(x):
    Sigma2=np.full((D,D), 0) # np.identity(D)
    np.fill_diagonal(Sigma2, 1.)
    """ choose mean and covariance of multivar gaussian proposal """
    return multivar_gaussian_proposal(x, cov=Sigma2) 

samplesMH2, weightsMH2 = run_mh(logP_1, T1, x_0mh, N)
print('MH2 (uncalibrated) Sample Size: ', len(samplesMH2),',','Acc. Ratio:',len(weightsMH2)/len(samplesMH2))
start = time.time()
print(acor.acor(samplesMH2.T))
tausMH2 = st.acor_calc(samplesMH2)
print("Process time/seconds: ", time.time()-start)

# Configure Triangle plots using 'getdist'
frac = 0.50     # Choose fraction of data points to ignore
b_ind_ADS = int(len(samples)*frac)
b_ind_em = int(len(samples2)*frac)
b_ind_mulem = int(len(samples3)*frac)
b_ind_MH = int(len(samplesMH)*frac)
b_ind_MH2 = int(len(samplesMH2)*0.02)
names = ["x%s"%i for i in range(D)]
labels =  ["x_%s"%i for i in range(D)]
corr_samples = MCSamples(samples=samples[b_ind_ADS:], names=names, labels=labels)
MH_samples = MCSamples(samples=samplesMH[b_ind_MH:], names=names, labels=labels)
MH_samples2 = MCSamples(samples=samplesMH2[b_ind_MH2:], names=names, labels=labels)
emcee_samples = MCSamples(samples=samples2[b_ind_em:], names=names, labels=labels)
multiemcee_samples = MCSamples(samples=samples3[b_ind_mulem:], names=names, labels=labels)

# Create random samples from the trial gaussian distribution to compare with MH ensemble
true_s = np.random.multivariate_normal(mu, Sigma, int(len(samples)-b_ind_ADS))       # Gaussian samples to compare with MH output 

# Plot estimated mean against number of likelihood calls
max_lhood_call_ADS = ms.lhood_calls(samples, weights, mu, sample_error=0.1)
max_lhood_call_emcee = ms.lhood_calls(samples2, weights2, mu, sample_error=0.1)
max_lhood_call_MH = ms.lhood_calls(samplesMH, np.ones(len(samplesMH)), mu, sample_error=0.1)
max_lhood_call_multidim = ms.lhood_calls(samples3, weights3, mu, sample_error=0.1)
print('Max # of calls to get mean within sample error:','\n','ADS:',max_lhood_call_ADS,'\n','emcee:',max_lhood_call_emcee,'\n','Multidim emcee',max_lhood_call_multidim,'\n','MH',max_lhood_call_MH)

# Set Latex font for figures
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

# Histogram of walker weights
fig1, ax1 = plt.subplots()
bins = np.arange(1,20) - 0.5
ax1.hist([weights[b_ind_ADS:], weights2[b_ind_em:], weights3[b_ind_mulem:]], bins, normed=True, alpha=0.5, label=['ADS','emcee','MDe'])
ax1.set_xlabel('Sample Weights')
ax1.set_ylabel('Normalised Walker Density')
ax1.set_xlim([0.5,5.5])
#plt.title(r'Histogram of Walker Weights ($\Sigma_{ij} = 1.8$))', fontsize=16)
plt.legend(loc='upper right')
fig1.savefig('hist_rosen_v2.pdf')

## Create Histogram of estimated means from individual ensemble walkers 
#walker_means = ms.sub_chain(samples[b_ind_ADS:], walk_identity[b_ind_ADS:], S) 
#fig2, ax2 = plt.subplots()
#bins = np.linspace(-0.75, 0.75, num=20)
#ax2.hist([walker_means], bins=bins, normed=False, label=['Individual walker means'])
#plt.legend(loc='upper right')
#fig2.savefig('walker_hist.pdf')
#
## Plot walker average RMS displacement and velocity
#ms.mean_walker_plot(mean_samples, 'temp_corr')
#
## Plot Vrms for two different distributions
#_,_,v_rms_av_corr,n_corr,_= ms.rms(mean_samples)          # Output Vrms for correlated MH Samples
#t_corr = np.arange(len(v_rms_av_corr))
#fig2, ax2 = plt.subplots()
#ax2.plot(t_corr,v_rms_av_corr, 'b', label='Vrms Correlated')
#ax2.legend()
#plt.savefig('vrms_temp.pdf')

# Triangle plot comparing true samples with MH output
plt.rcParams['text.usetex']=True
corr_true = MCSamples(samples=true_s, names=names, labels=labels, ranges={'x0':[-1, 1],'x1':[-1, 1],'x2':[-1, 1],'x3':[-1, 1]})
g = plots.getSubplotPlotter()
g.triangle_plot([corr_samples, MH_samples2, emcee_samples], filled=True, colors=['blue','green','red'], legend_labels=['ADS','MH Sph.','emcee'], legend_loc='upper right',legend_fontsize=16)
g.export('triangle_plot_rosen.pdf')
#g.triangle_plot([multiemcee_samples, emcee_samples], filled=True, colors=['green','red'], legend_labels=['MD emcee','emcee'], legend_loc='upper right',legend_fontsize=12)
#g.export('triangle_plots/gaussian/triangle_plot_emcee_1pt98.pdf')

# Print weighted covariance and mean of walker ensemble
w_average = np.average(samples[b_ind_ADS:], weights=weights[b_ind_ADS:], axis=0)
print('ADS Weighted Sample Mean:','\n', ms.vector_stats(w_average))
w_covariance = np.cov(samples[b_ind_ADS:], aweights=weights[b_ind_ADS:], rowvar=0) 
print('ADS Weighted Sample Covariance:')
#print(w_covariance)
print(ms.matrix_stats(w_covariance))

w_average = np.average(samples2[b_ind_em:], weights=weights2[b_ind_em:], axis=0)
print('emcee Weighted Sample Mean:','\n', ms.vector_stats(w_average))
w_covariance = np.cov(samples2[b_ind_em:], aweights=weights2[b_ind_em:], rowvar=0) 
print('emcee Weighted Sample Covariance:')
#print(w_covariance)
print(ms.matrix_stats(w_covariance))

w_average = np.average(samplesMH[b_ind_MH:], weights=np.ones(len(samplesMH[b_ind_MH:])), axis=0)
print('MH (cal) Weighted Sample Mean:','\n', ms.vector_stats(w_average))
w_covariance = np.cov(samplesMH[b_ind_MH:], aweights=np.ones(len(samplesMH[b_ind_MH:])), rowvar=0) 
print('MH (cal) Weighted Sample Covariance:')
#print(w_covariance)
print(ms.matrix_stats(w_covariance))

w_average = np.average(samplesMH2[b_ind_MH2:], weights=np.ones(len(samplesMH2[b_ind_MH2:])), axis=0)
print('MH (uncal) Weighted Sample Mean:','\n', ms.vector_stats(w_average))
w_covariance = np.cov(samplesMH2[b_ind_MH2:], aweights=np.ones(len(samplesMH2[b_ind_MH2:])), rowvar=0) 
print('MH (uncal) Weighted Sample Covariance:')
#print(w_covariance)
print(ms.matrix_stats(w_covariance))

w_average = np.average(samples3[b_ind_mulem:], weights=weights3[b_ind_mulem:], axis=0)
print('Multidim emceee Weighted Sample Mean:','\n', ms.vector_stats(w_average))
w_covariance = np.cov(samples3[b_ind_mulem:], aweights=weights3[b_ind_mulem:], rowvar=0) 
print('Multidim emceee Weighted Sample Covariance:')
#print(w_covariance)
print(ms.matrix_stats(w_covariance))

true_mu_s = np.mean(true_s.T, axis=1)
#print('Gaussian Sample Mean:','\n',true_mu_s)
true_cov_s = np.cov(true_s.T)
print('Gaussian Correlated Sample Covariance:')
#print(w_covariance)
print(ms.matrix_stats(true_cov_s))
