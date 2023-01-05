import numpy as np
import batman
import catwoman

## models used in step 1 ##
def init_batman_transit(theta, times):
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = theta[0]                        #time of inferior conjunction
    params.per = 10.**theta[1]                       #orbital period
    params.rp = theta[4]                       #planet radius (in units of stellar radii)
    params.a = 10.**theta[2]                        #semi-major axis (in units of stellar radii)
    params.inc = np.arccos(theta[3])*(180./np.pi)                      #orbital inclination (in degrees)
    params.ecc = 0. #theta[5]                       #eccentricity
    params.w = 90.#theta[6]                        #longitude of periastron (in degrees)
    params.u = [theta[5], theta[6]]      #limb darkening coefficients [u1, u2, u3, u4]
    #    LD model choice comes in as a float, so decode it
    #    1 = linear, 2 = quadratic, 0.5 = square root, 2.7 = logarithmic
    if theta[7] == 2.7:
        LDmodel = 'logarithmic'
    params.limb_dark = LDmodel       #limb darkening model

    init_batman_env = batman.TransitModel(params, times)    #initializes model
    return init_batman_env

def batman_transit(theta, model_initialization):
    """
    Computes a transit lightcurve from the 'batman' package
    Input params needed: tc in [day], P in [day], ...
    """
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = theta[0]                        #time of inferior conjunction
    params.per = 10.**theta[1]                       #orbital period
    params.rp = theta[4]                       #planet radius (in units of stellar radii)
    params.a = 10.**theta[2]                        #semi-major axis (in units of stellar radii)
    params.inc = np.arccos(theta[4])*(180./np.pi)                      #orbital inclination (in degrees)
    params.ecc = 0.#theta[5]                       #eccentricity
    params.w = 90.#theta[6]                        #longitude of periastron (in degrees)
    params.u = [theta[5], theta[6]]      #limb darkening coefficients [u1, u2, u3, u4]
    #    LD model choice comes in as a float, so decode it
    #    1 = linear, 2 = quadratic, 0.5 = square root, 2.7 = logarithmic
    if theta[7] == 2.7:
        LDmodel = 'logarithmic'
    params.limb_dark = LDmodel       #limb darkening model
    
    lc = model_initialization.light_curve(params)
    return lc
    

    