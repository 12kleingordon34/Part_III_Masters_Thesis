# Part III Masters Thesis
Code written for my Master's Thesis (Physical Natural Sciences, Cambridge)

## Contents
* `autocorr.py`: Contains autocorrelation functions for analysing Markov Chain independence criteria.

* `./ligo`: Contains source code used to construct LIGO likelihoods from simulations of Binary Black Hole mergers and LIGO detectors.
* `ligo.py`: Runs MCMC samplers on Euclidean and non-Euclidean LIGO likelihood simulations.
* `ligo_likelihood.py`: Stores the initialisation parameters for the LIGO data.

* `./metropolis_hastings`: Source code for Metropolis Hastings samplers
* `mh_sampler.py`: Runs Metropolis-Hastings functions on simulated likelihoods. Uses source code contained in `./metropolis_hastings`.

* `rosen.py`: Runs MCMC samplers on Rosenbrock benchmark dataset.

* `toroidal.py`: Run MCMC samplers on bimodal likelihood on toroidal manifolds.
* `spherical.py`: Run MCMC samplers on multimodal likelihood on spherical manifolds.

* `misc.py`: Contains functions to assist plotting and processing of results.
* `stepper.py`: Functions to implement stepping direction for MCMC samplers.
* `sph_tools.py`: Functions used to calculate MCMC sampler steps along geodesics on non-Euclidean likelihood surfaces.
