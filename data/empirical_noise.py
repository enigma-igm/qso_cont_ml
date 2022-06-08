'''Module for extracting real BOSS noise vectors and using them for synthetic spectra.'''

import numpy as np
import torch
import h5py
import os
from data.load_data import normalise_spectra
from qso_fitting.data.utils import rebin_spectra
import astropy.constants as const
from scipy.interpolate import interp1d
from IPython import embed

def loadSpectraBOSS(zmin, zmax):
    '''
    Load the BOSS spectra and all information relevant for prepping the empirical noise vectors, and convert the
    wavelength grids to rest-frame wavelengths.
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
    print ("Shape of wave_obs: {}".format(wave_obs.shape))
    flux_obs = np.copy(dset["spec"][grab]["flux"])
    sigma = np.copy(dset["spec"][grab]["sig"])
    redshift = np.copy(dset["meta"][grab]["Z_PIPE"])

    # close the file
    f.close()

    # convert the observed wavelengths to rest-frame wavelengths
    # has to be done row-wise
    wave_rest = np.zeros_like(wave_obs)
    for i in range(len(wave_obs)):
        wave_rest[i] = wave_obs[i] / (1 + redshift[i])

    return wave_rest, flux_obs, sigma, redshift


def interpBadPixels(wave_grid, ivar, gpm):
    '''
    Interpolate over (inverse variance) noise vector in bad pixels, using the information of the good pixels.
    We also interpolate over pixels where ivar < 0.

    @param wave_grid: ndarray of shape (n_wav,)
    @param ivar:
    @param gpm:
    @return:
    '''

    new_ivar = np.copy(ivar)

    for i in range(len(ivar)):

        nan_ivar = ~np.isfinite(ivar[i])
        neg_ivar = ivar[i] <= 0
        bad = nan_ivar | neg_ivar | ~gpm[i]
        interpolator = interp1d(wave_grid[~bad], ivar[i][~bad], kind="cubic", axis=-1,
                                bounds_error=False, fill_value="extrapolate")
        new_ivar[i][~bad] = interpolator(wave_grid[~bad])

    return new_ivar


def prepNoiseVectors(zmin, zmax):
    '''
    Load the BOSS spectra and prep them such that the noise vectors can be used for the synthetic spectra. The noise
    vectors returned by this function have non-zero noise everywhere, and the wavelengths are in the rest frame. The
    noise and spectra are also normalised to 1 at 1280 Angstrom. However, no regridding has been done yet.

    @param zmin: float
    @param zmax: float
    @param wave_min: float
    @param wave_max: float
    @param dloglam: float
    @return:
    '''

    # load the relevant data on the real spectra
    wave_rest, flux_obs, sigma, redshift = loadSpectraBOSS(zmin, zmax)

    # impute missing data by interpolation
    '''
    c_light = (const.c.to("km/s")).value
    dvpix = dloglam * c_light * np.log(10)
    wave_rest =
    '''
    for i in range(len(wave_rest)):

        nan_noise = ~np.isfinite(sigma[i])
        neg_noise = (sigma[i] <= 0)
        bad_wav = (wave_rest[i] <= 0)
        gpm = ~neg_noise & ~bad_wav & ~nan_noise
        interpolator = interp1d(wave_rest[i][gpm], sigma[i][gpm], kind="cubic", axis=-1, bounds_error=False,
                                fill_value="extrapolate")
        sigma[i][~gpm] = interpolator(wave_rest[i][~gpm])

    # make a good pixel mask indicating where the wavelength is > 0
    gpm = np.zeros_like(wave_rest, dtype=bool)
    for i in range(len(wave_rest)):
        gpm[i] = (wave_rest[i] > 0)

    # normalise the noise vectors
    flux_norm, sigma_norm = normalise_spectra(wave_rest, flux_obs, sigma)

    return wave_rest, flux_norm, sigma_norm, redshift, gpm


def rebinNoiseVectors(zmin, zmax, new_wave_grid):
    '''
    Load empirical BOSS noise vectors and rebin them onto the desired grid. The noise in bad pixels is interpolated
    over.

    @param zmin: float
    @param zmax: float
    @param new_wave_grid: ndarray of shape (n_pixels,)
    @return:
    '''

    # load the data and convert noise vectors to inverse variance vectors
    wave_rest, flux_norm, sigma_norm, redshift, gpm = prepNoiseVectors(zmin, zmax)
    ivar = 1 / sigma_norm**2

    # rebin onto the grid of the desired grid
    flux_rebin, ivar_rebin, gpm_rebin, count_rebin = rebin_spectra(new_wave_grid, wave_rest, flux_norm,
                                                                   ivar, gpm)

    # interpolate bad noise values
    for i in range(len(ivar_rebin)):

        interpolator = interp1d(new_wave_grid[gpm_rebin[i]], ivar_rebin[i][gpm_rebin[i]], kind="cubic",
                                bounds_error=False, fill_value="extrapolate")
        ivar_rebin[i][~gpm_rebin[i]] = interpolator(new_wave_grid[~gpm_rebin[i]])

    return flux_rebin, ivar_rebin, gpm_rebin