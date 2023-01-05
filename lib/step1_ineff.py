## ============================================================================ ##
## Matthew Murphy -- Last updated 12/28/2022
## This runs step 1 of the FALL code
##    and is an INEFFICIENT initial implementation of it ...
## This step takes in a normalized broadband light curve and
##    fits it, with models chosen herein, and related things prepped herein
## =========================================================================== ##
## -- Import modules
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import stats_functions as stats
import pickle
from multiprocessing import Pool
import corner
import emcee

## -- Defining paths for saving / loading
# present work directory
workdir = '..'
# path to FALL step 0 data to load in
inputdatafile = workdir+'/step0/outputs/step0_data.object'
# path to where this step's (step 1) outputs should be saved
outputpath = workdir+'/step1/'
if not os.path.isdir(outputpath):
    os.makedirs(outputpath)
    os.makedirs(outputpath+'figures/')
    os.makedirs(outputpath+'outputs/')

## -- Something to get the data ...
## can define own, or use pickled output from step 0 of FALL
with open(inputdatafile, 'rb') as inputfile_opened_binary:
    data = pickle.load(inputfile_opened_binary)
# add a time offset if it's missing w.r.t. step 0 and literature
time_offset = 2400000.5
data.times += time_offset
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
c = 1./np.log(10.)
#!# For testing purposes, I'll use GJ 3470b
# Transit parameters:
t0, t0unc = 2455953.6630, 0.0035
P, Punc = 3.336649, 0.000084
log10P, log10Punc = np.log10(P), ((c*Punc)/P)
aRs, aRsunc = 12.92, 0.72
log10aRs, log10aRsunc = np.log10(aRs), ((c*aRsunc)/aRs)
inc, incunc = 89.13, 0.34
cosi, cosiunc = np.cos(inc*(np.pi/180.)), np.sin(inc*(np.pi/180.))*(incunc*(np.pi/180.))
rprs, rprsunc = 0.07642, 0.00037
LDlaw = 'logarithmic'
LDcoeffs = [0.5, 0.5]
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
    transit_params = np.copy(theta[0:7])
    # Append LD law choice to end of parameter array
    #    needs to be a float, so use encoding
    #    1 = linear, 2 = quadratic, 0.5 = square root, 2.7 = logarithmic
    if LD_model_choice == 'logarithmic':
        LDcode = 2.7
    transit_params = np.append(transit_params, LDcode)
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
    lnLikelihood = stats.logLikelihood(data.norm_BBflux, data.norm_BBerr, full_model)
    # compute the posterior
    lnPost = lnLikelihood + lnPrior
    
    return lnPost
                     
    

## -- Initializing transit model
init_transit_params = get_transit_params(theta_init, LDlaw)
init_batman_env = init_batman_transit(init_transit_params, data.times)

## -- Checking the initial
plot_initial = True
show_plot_initial = False
init_tran_model = get_transit_model(init_transit_params, init_batman_env)
init_sys_model = get_sys_model(get_sys_params(theta_init), data.times)
init_fullmodel = init_tran_model * init_sys_model

init_lnPrior = logPrior(theta_init, mcmc_priors, mcmc_priorerrs)
init_lnLike = stats.logLikelihood(data.norm_BBflux, data.norm_BBerr, init_fullmodel)
init_lnPost = logPosterior(theta_init, data, mcmc_priors, mcmc_priorerrs, init_batman_env)
init_Chi2 = stats.computeChi2(data.norm_BBflux, data.norm_BBerr, init_fullmodel)
init_reducedChi2 = init_Chi2 / (len(data.times) - len(theta_init))
init_residuals = data.norm_BBflux - init_fullmodel

print('=============================')
print('Initial statistics')
print('   log Prior = %f'%(init_lnPrior))
print('   log Likelihood = %f'%(init_lnLike))
print('   log Posterior = %f'%(init_lnPost))
print('   chi^2 = %f'%(init_Chi2))
print('   reduced chi^2 = %f'%(init_reducedChi2))
print('   mean data uncertainty = %.0f ppm'%(np.mean(data.norm_BBerr)*1.e6))
print('   mean |residual| = %.0f ppm'%(np.mean(abs(init_residuals))*1.e6))
if plot_initial:
    fig, ax = plt.subplots(figsize=(10,10), nrows=3, sharex=True)
    # top panel: plot raw data and models
    ax[0].scatter(data.times, data.norm_BBflux, marker='o', s=3, c='black')
    ax[0].plot(data.times, init_tran_model, c='orange', label='Transit Model')
    ax[0].plot(data.times, init_sys_model, c='blue', label='Sys. Model')
    ax[0].plot(data.times, init_fullmodel, c='green')
    ax[0].set(ylabel='rel. brightness')
    ax[0].legend(loc='best')
    # middle panel: plot detrended data and b.f. astro model
    det_y, det_yerr = (data.norm_BBflux / init_sys_model), (data.norm_BBerr / init_sys_model)
    ax[1].scatter(data.times, det_y, marker='o', s=3, c='black', label='Detrended Data')
    ax[1].plot(data.times, init_tran_model, c='orange', label='Transit Model')
    ax[1].set(ylabel='rel. brightness')
    ax[1].legend(loc='best')
    # bottom panel: plot residuals
    ax[2].axhline(0., c='black', alpha=0.4)
    ax[2].scatter(data.times, init_residuals*1.e6, marker='o', c='black')
    ax[2].set(ylabel='residuals [ppm]', xlabel='time')
    plt.savefig(outputpath+'figures/initial_model.png', bbox_inches='tight')
    if show_plot_initial:
        plt.show()
    else:
        plt.close()
    
    
## -- Setting up my MCMC stuff    
Ndimensions = len(theta_init)
Nwalkers = 3*Ndimensions
burn = 100
Nsteps = 500 + burn
rerun = True      # re-runs MCMC in this iteration
saveVals = True  # saves chains to file
storeVals = False # stores best fit array to ipython cache
loadChains = not rerun # set True if not rerunning, will skip MCMC and laod in previously run chains instead

## -- Initializing walker positions
pos = np.zeros((Nwalkers, Ndimensions))
use_PrevVals = False # initialize wlakers based on posteriors of a prev. run

if use_PrevVals:
    ...
else:
    # if not, then use theta_init array
    for i in range(Ndimensions):
        pos[:,i] = theta_init[i] + 0.5*np.random.normal(0., mcmc_priorerrs[i], Nwalkers)
    print('MCMC walkers Initialized from scratch')

## -- Running MCMC
if rerun:
    with Pool() as pool: 
        # Collect additional arguments to the posterior function
        samp_args = (data, mcmc_priors, mcmc_priorerrs, init_batman_env)
        # Initialize and run sampler
        sampler = emcee.EnsembleSampler(Nwalkers, Ndimensions, logPosterior, pool=pool, args=samp_args)
        sampler.run_mcmc(pos, Nsteps, progress=True);
        
## -- Collecting the results
if rerun:
    # If we just re-ran the sampling, get those samples
    samples = sampler.get_chain()
    flatsamples = sampler.get_chain(flat=True)
    #loglikelihoods = sampler.get_log_prob(flat=True)
    #autocorrtimes = sampler.get_autocorr_time()

    samples = samples[burn:]
    flatsamples = flatsamples[burn*Nwalkers:]
    #loglikelihoods = loglikelihoods[burn*Nwalkers:]
    if saveVals:
        import h5py
        # save 'samples' and 'flatsamples' to an hdf5 file
        chain_savepath = outputpath+'outputs/'
        with h5py.File(chain_savepath+'samples.h5', 'w') as s_hf:
            s_hf.create_dataset('samples', data=samples)
        with h5py.File(chain_savepath+'flatsamples.h5', 'w') as f_hf:
            f_hf.create_dataset('flatsamples', data=samples)
else:
    # If not, load them in from wherever they live
    chain_loadpath = outputpath+'outputs/'
    import h5py
    with h5py.File(chain_loadpath+'samples.h5', 'r') as s_hf:
        samples = s_hf['samples'][:]
    with h5py.File(chain_loadpath+'flatsamples.h5', 'r') as f_hf:
        flatsamples = f_hf['flatsamples'][:]
    
# Computing the median of each parameter's sampling distribution as our 'best fit'
param_fits = np.asarray([np.median(flatsamples[:,i]) for i in range(samples.shape[2])])
# Computing the 16th and 84th percentile of each distribution, as an estimate for the 1-sigma uncertainty
param_uperrs = np.asarray([np.percentile(flatsamples[:,i], 84) for i in range(samples.shape[2])]) - param_fits
param_loerrs = param_fits - np.asarray([np.percentile(flatsamples[:,i], 16) for i in range(samples.shape[2])])
param_avgerrs = np.mean((param_uperrs, param_loerrs), axis=0)
# Defining labels for each paramter, for easy user intepretation of printed output
param_labels = np.array(['t0', 'log10P', 'log10aRs', 'cosi', 'rprs', 'u1', 'u2', 'sys a1', 'sys a0'])

print('Parameter Sampling Results:')
for i, param_name in enumerate(param_labels):
    print(param_name, ': ')
    print('Best Fit = ', param_fits[i], ' +/- ', param_avgerrs[i])
    print('   Uncertainties are + ', param_uperrs[i], ' , - ', param_loerrs[i])
    print('   Initial value was = ', theta_init[i])

## -- Set whether to plot (i.e. save) and show plots
# plots of light curve and best fit models
plot_final = True
show_plot_final = False
# plots of walker chains
plot_chains = True
show_plot_chains = False
# corner plot of posterior distributions
plot_corner = True
show_plot_corner = False

# compute median posterior models
bf_tran_model = get_transit_model(get_transit_params(param_fits, 'logarithmic'), init_batman_env)
bf_sys_model = get_sys_model(get_sys_params(param_fits), data.times)
bf_fullmodel = bf_tran_model * bf_sys_model
# compute detrended data
det_flux = data.norm_BBflux / bf_sys_model
det_err = data.norm_BBerr / bf_sys_model #!# should I save these to the data object??
# compute median posterior model statistics
bf_lnPrior = logPrior(param_fits, mcmc_priors, mcmc_priorerrs)
bf_lnLike = stats.logLikelihood(data.norm_BBflux, data.norm_BBerr, bf_fullmodel)
bf_lnPost = logPosterior(param_fits, data, mcmc_priors, mcmc_priorerrs, init_batman_env)
bf_Chi2 = stats.computeChi2(data.norm_BBflux, data.norm_BBerr, bf_fullmodel)
bf_reducedChi2 = bf_Chi2 / (len(data.times) - len(param_fits))
bf_residuals = data.norm_BBflux - bf_fullmodel

print('=============================')
print('Median ("best-fit") model statistics')
print('   log Prior = %f'%(bf_lnPrior))
print('   log Likelihood = %f'%(bf_lnLike))
print('   log Posterior = %f'%(bf_lnPost))
print('   chi^2 = %f'%(bf_Chi2))
print('   reduced chi^2 = %f'%(bf_reducedChi2))
print('   mean data uncertainty = %.0f ppm'%(np.mean(data.norm_BBerr)*1.e6))
print('   mean detrended data uncertainty = %.0f ppm'%(np.mean(det_err)*1.e6))
print('   mean |residual| = %.0f ppm'%(np.mean(abs(bf_residuals))*1.e6))
if plot_final:
    fig, ax = plt.subplots(figsize=(10,10), nrows=3, sharex=True)
    # top panel: plot raw data and models
    ax[0].scatter(data.times, data.norm_BBflux, marker='o', s=3, c='black')
    ax[0].plot(data.times, bf_tran_model, c='orange', label='Transit Model')
    ax[0].plot(data.times, bf_sys_model, c='blue', label='Sys. Model')
    ax[0].plot(data.times, bf_fullmodel, c='green', label='Full best fit model')
    ax[0].set(ylabel='rel. brightness')
    ax[0].legend(loc='best')
    # middle panel: plot detrended data and b.f. astro model
    ax[1].scatter(data.times, det_flux, marker='o', s=3, c='black', label='Detrended Data')
    ax[1].plot(data.times, bf_tran_model, c='orange', label='Transit Model')
    ax[1].set(ylabel='rel. brightness')
    ax[1].legend(loc='best')
    # bottom panel: plot residuals
    ax[2].axhline(0., c='black', alpha=0.4)
    ax[2].scatter(data.times, bf_residuals*1.e6, marker='o', c='black')
    ax[2].set(ylabel='residuals [ppm]', xlabel='time')
    plt.savefig(outputpath+'figures/bestfitmodel.png', bbox_inches='tight')
    if show_plot_final:
        plt.show()
    else:
        plt.close()
        
if plot_chains:
    fig2, ax2 = plt.subplots(samples.shape[2], figsize=(12,15), sharex=True)
    for i in range(samples.shape[2]):
        axx = ax2[i]
        axx.plot(samples[:,:,i], "k", alpha=0.3)
        axx.set_xlim(0, len(samples))
        axx.set_ylabel(param_labels[i])
    axx.set_xlabel('steps')
    plt.savefig(outputpath+'figures/walker_chains.png', bbox_inches='tight')
    if show_plot_chains:
        plt.show()
    else:
        plt.close()
        
if plot_corner:
    fig3 = corner.corner(flatsamples, labels=param_labels)
    plt.savefig(outputpath+'figures/corner.png', bbox_inches='tight')
    if show_plot_corner:
        plt.show()
    else:
        plt.close()
        
        
## -- Determining the current epoch transit time
## First, compute the current t0 by propagating the best fit previous t0 forward
# median of time array, just as a number to estimate N_transits between past and current transits with
approx_currenttime = np.median(data.times)
# get best fit t0 and period
bf_t0, bf_t0err = param_fits[0], param_avgerrs[0]
bf_P = 10.**(param_fits[1])
bf_Perr = (bf_P * param_avgerrs[1])/c
# estimate N_transits between then and now
N_estimate = np.rint((approx_currenttime - bf_t0)/bf_P)
# estimate current t0
current_t0_est = bf_t0 + (N_estimate*bf_P)
current_t0err_est = np.sqrt(bf_t0err**2 + (N_estimate**2)*(bf_Perr**2))

## Second, estimate the current t0 by fitting a simple transit model to the detrended data determined above
fixed_tran_params = np.array([param_fits[1], param_fits[2], param_fits[3], param_fits[4], param_fits[5], param_fits[6]])
    # Append LD law choice to end of parameter array
    #    needs to be a float, so use encoding
    #    1 = linear, 2 = quadratic, 0.5 = square root, 2.7 = logarithmic
if LDlaw == 'logarithmic':
    LDcode = 2.7
fixed_tran_params = np.append(fixed_tran_params, LDcode)
                              

# maybe use LSM instead of MCMC here ... since it's only one parameter
import batman
def fit_t0(x, t0):
    ## x= time, t0 = transit time being fit for
    # pull in fixed transit parameters from the global variable set previously
    ftp = fixed_tran_params
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = t0                        #time of inferior conjunction
    params.per = 10.**ftp[0]                       #orbital period
    params.rp = ftp[3]                       #planet radius (in units of stellar radii)
    params.a = 10.**ftp[1]                        #semi-major axis (in units of stellar radii)
    params.inc = np.arccos(ftp[2])*(180./np.pi)                      #orbital inclination (in degrees)
    params.ecc = 0. #theta[5]                       #eccentricity
    params.w = 90.#theta[6]                        #longitude of periastron (in degrees)
    params.u = [ftp[4], ftp[5]]      #limb darkening coefficients [u1, u2, u3, u4]
    #    LD model choice comes in as a float, so decode it
    #    1 = linear, 2 = quadratic, 0.5 = square root, 2.7 = logarithmic
    if ftp[6] == 2.7:
        LDmodel = 'logarithmic'
    params.limb_dark = LDmodel       #limb darkening model

    modelenv = batman.TransitModel(params, x)
    lc = modelenv.light_curve(params)
    return lc

from scipy.optimize import curve_fit
popt, pcov = curve_fit(fit_t0, data.times, det_flux, p0=current_t0_est, sigma=det_err, 
                       absolute_sigma=True, bounds=(current_t0_est-bf_P, current_t0_est+bf_P))
perr = np.sqrt(np.diag(pcov))

current_t0_fit, current_t0err_fit = popt, perr

print('=================================')
print('Estimates for current epoch transit time...')
print('  Value propagated from best fit previous t0 and ephemeris:')
print('        t0 = %f +/- %f'%(current_t0_est, current_t0err_est))
print('  Value determined from l.s. fit:')
print('        t0 = %f +/- %f'%(current_t0_fit, current_t0err_fit))

t0_sigdiff = abs(current_t0_fit-current_t0_est)/np.sqrt(current_t0err_est**2 + current_t0err_fit**2)
print('  These values are %.2f - sigma apart'%(t0_sigdiff))