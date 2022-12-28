## ============================================================================ ##
## Matthew Murphy -- Last updated 12/28/2022
## This runs step 1 of the FALL code
##    and is an INEFFICIENT initial implementation of it ...
## This step takes in a normalized broadband light curve and
##    fits it, with models chosen herein, and related things prepped herein
## =========================================================================== ##
## -- Import modules
import numpy as np
import sys
import stats_functions as stats

## -- Something to get the data ...
data = ...

#!# --
#!# For purposes of this implementation, I won't worry about smart model selection
#!# --

## -- Choosing which models I'll use:
##   astrophysical: a BATMAN transit model
##   systematic: a polynomial, visit-long flux vs. time ramp3
#astro_models = ['batman_transit']
#sys_models = ['visit_polynomial']
from systematic_models import visit_polynomial
from astro_models import init_batman_transit, batman_transit

## -- Setting parameters and associated uncertainties
# Transit parameters:
t0, t0unc = ..., ...
P, Punc = ..., ...
log10P, log10Punc = np.log10(P), ((c*Punc)/P)
aRs, aRsunc = ..., ...
log10aRs, log10aRsunc = np.log10(aRs), ((c*aRsunc)/aRs)
inc, incunc = ..., ...
cosi, cosiunc = np.cos(inc*(np.pi/180.)), np.sin(inc*(np.pi/180.))*(incunc*(np.pi/180.))
rprs, rprsunc = ..., ...
LDlaw = 'logarithmic'
LDcoeffs = [..., ...]
# Systematic parameters:
a1, a1unc = 0., 1.e-3
a0, a0unc = 1.0, 1.e-3

## -- Setting up arrays
# indices: 0 = t0, 1 = log10(P), 2 = log10(a/Rs), 3 = cos(i), 4 = Rp/Rs, 5 = LD u1
#          6 = LD u2, 7 = sys a1, 8 = sys a0
theta_init = np.array([t0, log10P, log10aRs, cosi, rprs, LDcoeffs[0], LDcoeffs[1], a1, a0])
mcmc_priors = np.array([t0, log10P, log10aRs, cosi, np.inf, np.inf, np.inf, np.inf, np.inf])
mcmc_priorerrs = np.array([t0unc, log10Punc, log10aRsunc, cosiunc, 
                           1.e-4, 0.2, 0.2, 1.e-3, 1.e-3])

## -- Defining functions to use
def get_transit_params(theta, LD_model_choice):
    # Copy input parameters, which should have everything but the LD law
    transit_params = np.copy(theta)
    # Append LD law choice to end of parameter array
    transit_params = np.append(transit_params, LD_model_choice)
    # return
    return transit_params

def get_transit_model(params, initialized_env):
    transit_y = batman_transit(params, initialized_env)
    return transit_y

def get_sys_params(theta):
    # Pull out input params
    sys_params = np.array([theta[7], theta[8]])
    # return 
    return sys_params

def get_sys_model(params, x_arrays):
    poly_y = visit_polynomial(x_arrays, params, n=1)
    return poly_y

def logPrior(theta, priors, priorserr):
    # Enforce limits on certain parameters:
    # limit Rp/Rs to [0,1]
    if not (0. <= theta[4] <= 1.): return -np.inf
    # limit LD coeffs to [0,1]
    if not (0. <= theta[5] <= 1.): return -np.inf
    if not (0. <= theta[6] <= 1.): return -np.inf
    # Compute Gaussian priors when enforced
    lnP = 0.
    for i, priorval in enumerate(priors):
        if np.isinf(priorval):
            # If not applying a prior on a parameter, it's listed as inf in the array
            lnP += 0 #continue
        else:
            lnP += -(theta[i] - priorval) ** 2. / (2. * priorserr[i] ** 2.) - np.log(np.sqrt(2. * priorserr[i] **2. * np.pi))
    return lnP 
    
def logPosterior(theta, 
                 data, mcmc_priors, mcmc_priorerrs, batman_env):
    # Compute the prior value
    lnPrior = logPrior(theta, mcmc_priors, mcmc_priorerrs)
    # Compute astro model
    tran_params = get_transit_params(theta, 'logarithmic')
    tran_model = get_transit_model(tran_params, batman_env)
    # Compute systematic model
    sys_params = get_sys_params(theta)
    sys_model = get_sys_model(sys_params, data.times)
    # combine models
    full_model = tran_model * sys_model
    # compute the likelihood
    lnLikelihood = stats.logLikelihood(data.relBrights, data.relBrightUncs, full_model)
    # compute the posterior
    lnPost = lnLikelihood + lnPrior
    
    return lnPost
                     
    

## -- Initializing transit model
init_transit_params = get_transit_params(theta_init, LDlaw)
init_batman_env = init_batman_transit(init_transit_params, data.times)

## -- Checking the initial
plot_initial = True
init_tran_model = get_transit_model(init_transit_params, init_batman_env)
init_sys_model = get_sys_model(get_sys_params(theta_init), data.times)
init_fullmodel = init_tran_model * init_sys_model

init_lnPrior = logPrior(theta_init, mcmc_priors, mcmc_priorerrs)
init_lnLike = stats.logLikelihood(data.relBrights, data.relBrightUncs, init_fullmodel)
init_lnPost = logPosterior(theta_init, data, mcmc_priors, mcmc_priorerrs, init_batman_env)
init_Chi2 = stats.computeChi2(data.relBrights, data.relBrightUncs, init_fullmodel)
init_reducedChi2 = init_Chi2 / (len(data.times) - len(theta_init))
init_residuals = data.relBrights - init_fullmodel

print('=============================')
print('Initial statistics')
print('   log Prior = %f'%(init_lnPrior))
print('   log Likelihood = %f'%(init_lnLike))
print('   log Posterior = %f'%(init_lnPost))
print('   chi^2 = %f'%(init_Chi2))
print('   reduced chi^2 = %f'%(init_reducedChi2))
print('   mean data uncertainty = %.0f ppm'%(np.mean(data.relBrightUncs)*1.e6))
print('   mean |residual| = %.0f ppm'%(np.mean(abs(init_residuals))*1.e6))
if plot_initial:
    fig, ax = plt.subplots(figsize=(10,10), nrows=3, sharex=True)
    # top panel: plot raw data and models
    ax[0].scatter(data.times, data.relBrights, marker='o', s=3, c='black')
    ax[0].plot(data.times, init_tran_model, c='orange', label='Transit Model')
    ax[0].plot(data.times, init_sys_model, c='blue', label='Sys. Model')
    ax[0].plot(data.times, init_fullmodel, c='green')
    ax[0].set(ylabel='rel. brightness')
    ax[0].legend(loc='best')
    # middle panel: plot detrended data and b.f. astro model
    det_y, det_yerr = (data.relBrights / init_sys_model), (data.relBrightsUncs / init_sys_model)
    ax[1].scatter(data.times, det_y, marker='o', s=3, c='black', label='Detrended Data')
    ax[1].plot(data.times, init_tran_model, c='orange', label='Transit Model')
    ax[1].set(ylabel='rel. brightness')
    ax[1].legend(loc='best')
    # bottom panel: plot residuals
    ax[2].axhline(0., c='black', alpha=0.4)
    ax[2].scatter(data.times, init_residuals*1.e6, marker='o', c='black')
    ax[2].set(ylabel='residuals [ppm]', xlabel='time')
    plt.show()

