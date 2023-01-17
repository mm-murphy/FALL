import os
import numpy as np
from astropy.io import fits
from jwst.pipeline.calwebb_detector1 import Detector1Pipeline

def run1(filename):
    ##
    ##
    if not ('uncal.fits' in filename):
        print('error')
        return np.nan

    # Run step 1 of the jwst pipeline
    result1 = Detector1Pipeline.call(filename)


