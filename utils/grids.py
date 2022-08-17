'''Module containing utility functions for creating various wavelength grids.'''

import numpy as np
import astropy.constants as const

c_light = const.c.to("km/s").value

def get_wave_grid(wave_min, wave_max, dvpix, force='min'):
    """
    Taken from dw_inference.sdss.utils on July 11 2022.

    Get a wavelength grid with uniform velocity separation, and a split at wave_blu_red. Wavelengths
    can either be in rest-frame or observed frame
    Args:
        wave_min (float):
            Minimum wavelength
        wave_max (float):
            Maximum wavelength
        dvpix (float):
            Pixel separaiton in velocity units
        force (str):
            If 'min', will enforce the minimum to be exactly wave_min, if 'max' will enforce the maximum to be
            exactly wave_max.
    Returns:
        wave_grid:
            Wavelength grid
    """
    # Create the observed wavelength grid with hybrid fine-coarse velocity scale
    dloglam = dvpix/c_light/np.log(10.0)
    loglam_min = np.log10(wave_min)
    loglam_max = np.log10(wave_max)
    npix = int(np.round((loglam_max - loglam_min)/dloglam))
    # We enforce the maximum of the grid to always be loglam_max, and let the minimum absorb the slop
    # resulting from the integer number of pixels
    if force == 'min':
        loglam = loglam_min + np.arange(npix) * dloglam
    elif force == 'max':
        loglam = loglam_max - np.arange(npix)*dloglam
        loglam = loglam[::-1]
    else:
        raise ValueError('Unrecognized value for force')

    wave_grid = np.power(10.0, loglam)
    return wave_grid


def get_blu_red_wave_grid(wave_min, wave_max, wave_blu_red, dvpix_blu, dvpix_red):
    """
    Taken from dw_inference.simulator.utils on July 11 2022.

    Get a wavelength grid with uniform velocity separation, and a split at wave_blu_red. Wavelengths
    can either be in rest-frame or observed frame
    Args:
        wave_min (float):
           Minimum wavelength
        wave_max (float):
           Maximum wavelength
        wave_blu_red:
           Wavelength at which we switch from fine sampling (on the blue side) to coarse sampling (on the red side)
        dvpix_blu (float):
           Pixel scale on the blue side
        dvpix_red (float):
           Pixel scale on the red side
    Returns:
        wave_rest (ndarray):
            Rest wavelength grid
        dvpix_diff (ndarray):
            Velocity difference between pixels
        ipix_blu (ndarray):
            Boolean array which is true for blue side pixels
        ipix_red (ndarray):
            Boolean array which is true for red pixels
    """
    # Create the observed wavelength grid with hybrid fine-coarse velocity scale
    wave_blu = get_wave_grid(wave_min, wave_blu_red, dvpix_blu, force='max')
    loglam_red_min = np.max(np.log10(wave_blu)) + dvpix_red / c_light / np.log(10.0)
    wave_red_min = np.power(10.0, loglam_red_min)
    wave_red = get_wave_grid(wave_red_min, wave_max, dvpix_red, force='min')

    wave_grid = np.hstack([wave_blu, wave_red])
    # Indices for blue and red pixels (blue pixels are the forest, red pixels are used in the red-side fit)
    ipix_blu = np.hstack([np.ones_like(wave_blu, dtype=bool), np.zeros_like(wave_red, dtype=bool)])
    ipix_red = np.hstack([np.zeros_like(wave_blu, dtype=bool), np.ones_like(wave_red, dtype=bool)])

    # The velocity difference between consecutive pixels
    dvpix_diff = np.insert(np.diff(np.log10(wave_grid) * c_light * np.log(10.0)), 0, dvpix_blu)

    return wave_grid, dvpix_diff, ipix_blu, ipix_red


def rest_BOSS_grid(wave_min=1020., wave_max=1970., dloglam=1e-4):

    c_light = (const.c.to("km/s")).value
    dvpix = dloglam * c_light * np.log(10)

    wave_grid = get_wave_grid(wave_min, wave_max, dvpix)

    return wave_grid


def obs_BOSS_grid(redshift, wave_min=1020., wave_max=1970., dloglam=1e-4):

    wave_rest = rest_BOSS_grid(wave_min, wave_max, dloglam)
    wave_obs = (1 + redshift) * wave_rest

    return wave_obs


def obs_BOSS_grid2d(redshifts, wave_min=1020., wave_max=1970., dloglam=1e-4):

    wave_rest = rest_BOSS_grid(wave_min, wave_max, dloglam)

    wave_obs = np.zeros((len(redshifts), len(wave_rest)))
    for i,z in enumerate(redshifts):
        wave_obs[i] = (1 + z) * wave_rest

    return wave_obs