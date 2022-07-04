'''Module containing utility functions for creating various wavelength grids.'''

import numpy as np
from dw_inference.sdss.utils import get_wave_grid
from dw_inference.simulator.utils import get_blu_red_wave_grid
import astropy.constants as const


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