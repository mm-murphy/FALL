import numpy as np

def logLikelihood(ydata, yerr, modely):
    lnL = 0.
    chi_array = ((ydata - modely) ** 2. / yerr ** 2.) + np.log(2. * np.pi * yerr ** 2.)
    lnL += -0.5 * np.sum(chi_array)
    
    return lnL

def computeBIC(Nparams, Ndata, lnL):
    bic = Nparams*np.log(Ndata) - 2.*lnL
    return bic 

def computeChi2(ydata, yerr, modely):
    chi_array = ((ydata - modely) ** 2. / yerr ** 2.)
    return np.sum(chi_array)