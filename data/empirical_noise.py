'''Module for extracting real BOSS noise vectors and using them for synthetic spectra.'''

import numpy as np
import torch
import h5py
import os
from data.load_data import normalise_spectra

def loadNoiseVectors(zmin, zmax):
    '''
    Load empirical noise vectors from the BOSS spectra, and converting the wavelength grids to rest-frame wavelengths.
    The file containing the BOSS spectra should be named "IGMspec_DB_v03.1.hdf5" and should be stored in the
    environment "SPECDB".

    @param zmin: float
            Minimum wavelength to load spectra for, in Angstrom.
    @param zmax: float
            Maximum wavelength to load spectra for, in Angstrom.
    @return:
    '''

    datafile = os.getenv("SPECDB") + "/IGMspec_DB_v03.1.hdf5"
    f = h5py.File(datafile, "r")
    dset = f["BOSS_DR14"]

    # make a mask of which spectra to use
    grab = (dset["meta"]["ZWARNING"] == 0) & (dset["meta"]["Z_PIPE"] > zmin) & (dset["meta"]["Z_PIPE"] < zmax)

    # load the wavelength, flux and 1sigma noise vectors, and the redshifts
    wave_obs = np.copy(dset["spec"][grab]["wave"])
    flux_obs = np.copy(dset["spec"][grab]["flux"])
    sigma = np.copy(dset["spec"][grab]["sig"])
    redshift = np.copy(dset["meta"][grab]["Z_PIPE"])

    # close the file
    f.close()

    # convert the observed wavelengths to rest-frame wavelengths
    wave_rest = wave_obs / (1 + redshift)

    return wave_rest, flux_obs, sigma, redshift

