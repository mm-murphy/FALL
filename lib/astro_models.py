import numpy as np
import batman
import catwoman

def init_batman_transit(theta, times):
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = theta[0]                        #time of inferior conjunction
    params.per = theta[1]                       #orbital period
    params.rp = theta[2]                       #planet radius (in units of stellar radii)
    params.a = theta[3]                        #semi-major axis (in units of stellar radii)
    params.inc = theta[4]                      #orbital inclination (in degrees)
    params.ecc = theta[5]                       #eccentricity
    params.w = theta[6]                        #longitude of periastron (in degrees)
    params.limb_dark = theta[7]       #limb darkening model
    params.u = [theta[8]]      #limb darkening coefficients [u1, u2, u3, u4]

    init_batman_env = batman.TransitModel(params, times)    #initializes model
    return init_batman_env

def batman_transit(theta, model_initialization):
    """
    Computes a transit lightcurve from the 'batman' package
    Input params needed: tc in [day], P in [day], ...
    """
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = theta[0]                        #time of inferior conjunction
    params.per = theta[1]                       #orbital period
    params.rp = theta[2]                       #planet radius (in units of stellar radii)
    params.a = theta[3]                        #semi-major axis (in units of stellar radii)
    params.inc = theta[4]                      #orbital inclination (in degrees)
    params.ecc = theta[5]                       #eccentricity
    params.w = theta[6]                        #longitude of periastron (in degrees)
    params.limb_dark = theta[7]       #limb darkening model
    params.u = [theta[8]]      #limb darkening coefficients [u1, u2, u3, u4]
    
    lc = model_initialization.light_curve(params)
    return lc
    
    