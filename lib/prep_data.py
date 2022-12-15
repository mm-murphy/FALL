import numpy as np

def normalize(data, data_err, method='median', norm_idxs='all'):
    
    if norm_idxs == 'all':
        y_tocalc = np.copy(data)
    elif type(norm_idxs) == np.array:
        y_tocalc = np.copy(data[norm_idxs])
        
    if method == 'median':
        norm_factor = np.median(y_tocalc)
    elif method == 'mean':
        norm_factor = np.mean(y_tocalc)
        
    normed_data = data / norm_factor
    normed_errs = data_err / norm_factor
    return normed_data, normed_errs, norm_factor

def prep_broadband(data_obj, **kwargs):
    """
    allowed kwargs: 'method' = normalization method (default is 'median'), 'norm_idxs' = indices of 
        data points to use in normalization method (default is 'all')
    """
    
    y, yerr = data_obj.raw_BBflux, data_obj.raw_BBerr
    if 'method' in kwargs:
        norm_method = kwargs['method']
    else:
        norm_method = 'median'
        
    if 'norm_idxs' in kwargs:
        norm_idxs = kwargs['norm_idxs']
    else:
        norm_idxs = 'all'
        
    norm_y, norm_yerr, norm_fac = normalize(y, yerr, norm_method, norm_idxs)
    data_obj.norm_BBflux = norm_y
    data_obj.norm_BBerr = norm_yerr
    data_obj.BBnormfactor = norm_fac
    
    return data_obj

def prep_spectral(data_obj, **kwargs):
    """
    allowed kwargs: 'method' = normalization method (default is 'median'), 'norm_idxs' = indices of 
        data points to use in normalization method (default is 'all')
    """
    
    y, yerr = data_obj.raw_TSflux, data_obj.raw_TSerr
    waves = data_obj.wavelengths
    if 'method' in kwargs:
        norm_method = kwargs['method']
    else:
        norm_method = 'median'
        
    if 'norm_idxs' in kwargs:
        norm_idxs = kwargs['norm_idxs']
    else:
        norm_idxs = 'all'
        
    norm_y, norm_yerr = np.copy(y), np.copy(yerr)
    normfacs = np.zeros(len(waves))
    for i_wav, wavelength in enumerate(waves):
        this_y, this_yerr = y[i_wav,:], yerr[i_wav,:]
        this_normy, this_normerr, this_normfac = normalize(this_y, this_yerr, norm_method, norm_idxs)
        
        norm_y[i_wav] = this_normy
        norm_yerr[i_wav] = this_normerr
        normfacs[i_wav] = this_normfac
        
    data_obj.norm_TSflux = norm_y
    data_obj.norm_TSerr = norm_yerr
    data_obj.TSnormfactor = normfacs
    
    return data_obj
    
    
    
    