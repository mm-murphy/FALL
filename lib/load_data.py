import numpy as np
import h5py

class DATA:
    """
    Class which holds light curve data
    Args:
        times, raw_BBflux, raw_BBerr, raw_TSflux, raw_TSerr, wavelengths
    """
    def __init__(self, times, raw_BBflux, raw_BBerr, raw_TSflux, raw_TSerr, wavelengths):
        self.times = times
        self.raw_BBflux = raw_BBflux
        self.raw_BBerr = raw_BBerr
        self.raw_TSflux = raw_TSflux
        self.raw_TSerr = raw_TSerr
        self.wavelengths = wavelengths


def load_data(data_file_path):
    data_load = h5py.File(data_file_path, 'r')
    
    times = np.array(data_load['time'])
    raw_bbflux = np.array(data_load['flux_white'])
    raw_bberr = np.array(data_load['err_white'])
    raw_tsflux = np.array(data_load['data'])
    raw_tserr = np.array(data_load['err'])
    waves = np.array(data_load['wavelength'])
    data_load.close()

    data = DATA(times, raw_bbflux, raw_bberr, raw_tsflux, raw_tserr, waves)
    return data
