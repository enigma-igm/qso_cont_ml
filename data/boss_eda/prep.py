''''Module for loading the redshifts and SDSS ugriz PSF magnitudes, and computing the corresponding logLv values.'''

import numpy as np
import h5py
from enigma.tpe.m912 import m912_mags
import astropy.constants as const
import os
import pkg_resources
import pickle

def deredden(psfmags, extinction):
    '''
    Dereddens the magnitudes for the 5 SDSS filters (u, g, r, i, z).

    @param psfmags: ndarray of shape (n_qso, 5)
    @param extinction: ndarray of shape(n_qso, 5)
    @return:
        mags: ndarray of shape (n_qso, 5)
    '''

    assert psfmags.shape == extinction.shape
    assert psfmags.shape[-1] == 5

    mags = psfmags - extinction

    return mags


def getlogLv(mags, redshifts, cosmo=False, ALPHA_EUV=1.7):
    '''
    Compute the log of the luminosity at the Lyman limit. Wrapper for m912_mags.

    @param mags: ndarray of shape (n_qso, 5)
        Dereddened magnitudes in the 5 SDSS ugriz filters.
    @param redshifts: ndarray of shape (n_qso,)
    @param cosmo: astropy cosmology FLRW instance
    @param ALPHA_EUV: float
    @return:
    '''

    assert len(redshifts) == mags.shape[0]

    _, logLv = m912_mags(redshifts, mags, cosmo=cosmo, ALPHA_EUV=ALPHA_EUV)

    return logLv


def loadBOSSmeta(wave_min, wave_max, SN_min, dloglam=1e-4, z_min=None, z_max=None):
    '''
    Extract the redshifts, ugriz PSF magnitudes and corresponding extinction of BOSS DR14 QSOs.
    Modeled after sdss_data() from qso_fitting.data.sdss.sdss.

    @param wave_min: float
    @param wave_max: float
    @param SN_min: float
    @param dloglam: float
    @param z_min: float or NoneType
    @param z_max: float or NoneType
    @return:
        redshifts: ndarray of shape (n_qso,)
        psfmags: ndarray of shape (n_qso, 5)
        extinction: ndarray of shape (n_qso, 5)
    '''

    # assume that BOSS covers 3650-10000 A
    BOSS_min = 3650.0
    BOSS_max = 1e4

    # avoid edge effects by adding/subtracting a bit
    wave_cut_min = np.power(10., np.log10(wave_min) - 1.5 * dloglam)
    wave_cut_max = np.power(10., np.log10(wave_max) + 1.5 * dloglam)

    # If no redshift range is specified, use a range that guarantees coverage of the requested range of wavelengths.
    z_min_use = BOSS_min / (wave_cut_min - 10.) - 1. if z_min is None else z_min
    z_max_use = BOSS_max / (wave_cut_max + 10.) - 1. if z_max is None else z_max

    # load in the data file and the SNR file
    db_filename = os.getenv("SPECDB") + "/IGMspec_DB_v03.1hdf5"
    db_file = h5py.File(db_filename, "r")

    try:
        snr_file = pkg_resources.resource_filename("dw_inference.sdss", "dr14_median_snr.pckl")
    except TypeError:
        dwpath = os.getenv("dw_inference")
        snr_file = dwpath + "/sdss/dr14_median_snr.pckl"

    median_snr = pickle.load(open(snr_file, "rb"))

    # apply the redshift selection and redshift quality cut and filter out BALs
    zcut = (db_file["BOSS_DR14/meta"]["Z_PIPE"] > z_min_use) & (db_file["BOSS_DR14/meta"]["Z_PIPE"] < z_max_use) & \
           (db_file["BOSS_DR14/meta"]["ZWARNING"] == 0)
    bal_cut = db_file["BOSS_DR14/meta"]["BI_CIV"] > 0.
    not_bal = zcut & np.logical_not(bal_cut)
    print('There are {:d} quasars between z_min={:5.3f} and z_max={:5.3f}'.format(np.sum(zcut), z_min_use, z_max_use))
    print('Of these, {:d} quasars were flagged as BAL, resulting in a parent sample of {:d} quasars'.format(
        np.sum(zcut & bal_cut), np.sum(not_bal)))

    # also filter out low-SNR spectra
    grab = not_bal & (median_snr > SN_min * .5)

    # extract the ugriz PSF magnitudes and the corresponding extinction
    psfmags = np.array(db_file["BOSS_DR14/meta"][grab]["PSFMAG"])
    extinction = np.array(db_file["BOSS_DR14/meta"][grab]["GAL_EXT"])

    # also return the corresponding redshifts
    redshifts = np.array(db_file["BOSS_DR14/meta"][grab]["Z"])

    db_file.close()

    return redshifts, psfmags, extinction