'''Module for loading previously extracted BOSS DR14 redshifts and logLv values.'''

import numpy as np
import h5py
import os

def loadRedshiftLuminosityFile(savepath=None):
    '''
    Load the redshifts and log-luminosities at the Lyman limit of BOSS DR14 quasars.

    @param savepath: str or NoneType
    @return:
    '''

    if savepath is None:
        savepath = os.getenv("SPECDB")
    elif ~isinstance(savepath, str):
        raise TypeError("Argument 'savepath' must be a string or None.")

    f = h5py.File(savepath+"/luminosity-redshift-metadata.hdf5", "r")

    redshifts = np.array(f["redshifts"])
    logLv = np.array(f["logLv"])

    f.close()

    return redshifts, logLv