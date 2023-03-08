import numpy as np
from scipy.optimize import newton
def timetrans_to_timeperi(tc, per, ecc, omega):
    """
    Convert Time of Transit to Time of Periastron Passage
    Args:
        tc (float): time of transit
        per (float): period [days]
        ecc (float): eccentricity
        omega (float): longitude of periastron (radians)

    Returns:
        float: time of periastron passage
        
    Copied from source code <https://radvel.readthedocs.io/en/latest/_modules/radvel/orbit.html#timetrans_to_timeperi>

    """
    try:
        if ecc >= 1:
            return tc
    except ValueError:
        pass

    f = np.pi/2 - omega
    ee = 2 * np.arctan(np.tan(f/2) * np.sqrt((1-ecc)/(1+ecc)))  # eccentric anomaly
    tp = tc - per/(2*np.pi) * (ee - ecc*np.sin(ee))      # time of periastron

    return tp

def ecc_anomaly_eqn(EA, ecc, M):
    f = EA - ecc*np.sin(EA) - M
    return f

def true_anomaly(tc, P, par1, par2, times):
    # tc = transit time in [bjd tdb], P = period in [day]
    # par1 = sqrt(e)cos(w), par2 = sqrt(e)sin(w)
    
    ecc = par1**2 + par2**2     # eccentricity
    w = np.arctan2(par2, par1)  # argument of periapsis
    
    # Compute time of periastron passage
    tp = timetrans_to_timeperi(tc, P, ecc, w)
    
    # Compute mean anomalies
    n =  (2. * np.pi) / P
    M = n*(times - tp)
    
    # Root-find to solve for eccentric anomaly
    EA = np.asarray([])
    for Mval in M:
        EAval = newton(ecc_anomaly_eqn, 75., args=(ecc, Mval))
        EA = np.append(EA, EAval)
    
    # Compute projections of true anomaly
    cosf = (np.cos(EA) - ecc) / (1. - ecc*np.cos(EA))
    sinf = (np.sin(EA) * np.sqrt(1. - ecc*ecc)) / (1. - ecc*np.cos(EA))
    
    # Compute true anomaly
    f = np.arctan2(sinf, cosf)
    
    return f

def radial_velocity(rv_theta, times):
    
    # Unpack parameters
    tc = rv_theta[0]
    P = 10.**(rv_theta[1])
    par1 = rv_theta[2]      # sqrt(e) * cos(omega)
    par2 = rv_theta[3]      # sqrt(e) * sin(omega)
    K = rv_theta[4]/1000.   # RV semi-amplitude (km/s)
    gamma = rv_theta[5]     # RV system velocity offset (km/s)
    
    ecc = par1**2 + par2**2     # eccentricity
    w = np.arctan2(par2, par1)  # argument of periapsis
    
    
    f = true_anomaly(tc, P, par1, par2, times)
    
    arg = np.cos(w + f) + ecc*np.cos(w)
    v = K*arg + gamma
    
    return v