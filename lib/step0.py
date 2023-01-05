## ================================================ ##
## -- Step 0 -- Load in and prepare the data
from load_data import load_data
from prep_data import prep_broadband, prep_spectral
import plots
import os, sys
import pickle

def run_step0(datafile_path, workdir, makeplots=True, showplots=False, savedata=True):
    ## Inputs:
    ##   datafile_path = string containing full path of data file 
    ##        (currently, only hdf5 outputs of Eureka are supported)
    ##   workdir = string containing work directory (i.e. pwd)
    ##        within which we'll save data and figures
    ##   makeplots = boolean, if True we create and save figures
    ##   showplots = boolean, if True we show above plots to console
    ##   savedata = boolean, if True we save data object, modified herein, to file
    ##         in addition to just returning it
    ## ============================================================= ##
    # load the data into an object:
    dataload = load_data(datafile_path)
    # normalize the lightcurves
    data = prep_broadband(dataload, method='median', norm_idxs='all')
    data = prep_spectral(data, method='median', norm_idxs='all')
    # if desired, make plots to check whether it worked
    if makeplots:
        # specify path to save figures to
        #   and check if it exists, and create it if not
        s0_figpath = workdir+'/step0/figs'
        if not os.path.isdir(s0_figpath):
            os.makedirs(s0_figpath)
            print('Figure directory created at ', s0_figpath)
        else:
            print('Figures being saved at ', s0_figpath)
            
        plots.plot_0_a(data, s0_figpath, showplots)
        #plots.plot_0_b(data, s0_figpath, showplots) #!# commented this out for now while only focusing on broadband stuff
    # if desired, save the data object
    if savedata:
        # specify path to save data to
        #    and check if it exists, and create it if not
        s0_savepath = workdir+'/step0/outputs'
        if not os.path.isdir(s0_savepath):
            os.makedirs(s0_savepath)
            print('Output save directory created at ', s0_savepath)
        else:
            print('Saving outputs at ', s0_savepath)
        
        # save the data object using pickle
        savefile = s0_savepath+'/step0_data.object'
        with open(savefile, 'wb') as savefile_opened_binary:
            pickle.dump(data, savefile_opened_binary)
    # after all is said and done ... return the data object
    return data