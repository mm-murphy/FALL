import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
sys.path.append('lib')

## ================================================ ##
workdir = '.'
datadir = '/home/matthew/data/jwst/gj3470/nircam/TaylorBellAnalyses/'
datafile = datadir+'S4_GJ3470b_transit_F322W2_ap9_bg14_LCData.h5'

## ================================================ ##
## -- Step 0 -- Load in and prepare the data
# updated code:
import step0 as s0
s0_data = s0.run_step0(datafile, workdir, makeplots=True, showplots=False, savedata=True)

# old code:
##from load_data import load_data
##from prep_data import prep_broadband, prep_spectral
##import plots
##s0figpath = workdir+'/figs/step0/'
##
### Set pwd
### Set the path to the directory where the data is held
##data_dir = '/home/mmmurphy/data/JWST/gj3470/nircam/TaylorBellAnalyses/'
##data_file = data_dir + 'S4_GJ3470b_transit_F322W2_ap9_bg14_LCData.h5'
### Load it into an object
### See lib/load_data.py for the attributes
##data = load_data(data_file)
### Normalize the lightcurves
##data = prep_broadband(data, method='median', norm_idxs='all')
##data = prep_spectral(data, method='median', norm_idxs='all')
### make plots to check this worked !!
##plots.plot_0_a(data, s0figpath)
##plots.plot_0_b(data, s0figpath)


## =========================================== ##
## Step 1 -- Fit the broadband light curve

# pick which models to use
#   available astrophysical models are ['batman_transit']
#   available systematic models are ['visit_ramp1']
astro_models = ['batman_transit']
sys_models = ['visit_ramp1']
# define parameters to fit for
## -- see 'lib/step1_ineff.py' for an implementation 

## ========================================== ##
## Step 2 -- Fit the spectral light curves, assuming uniform limbs
